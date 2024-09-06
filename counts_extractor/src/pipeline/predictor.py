import os
import sys
import json
import argparse
import datetime
import itertools
import numpy as np
import pandas as pd
from joblib import load
from pathlib import Path
from src.logger import logging
from collections import defaultdict
from src.expection import CustomException


class CountsPredictor:

    def __init__(self, dir_path: str, location: str, output_file: str) -> None:
        """
        Initializes a new instance of the CountsPredictor class.

        Args:
            dir_path (str): The path to the directory containing the extracted counts for the two processed vidoes.
            location (str): The camera id for which the predictor is being initialized.
            output_file (str): The path to the output file.

        Raises:
            CustomException: If there is an error loading the scaler or model.

        Initializes the following instance variables:
            - scaler (object): The loaded scaler object for a specified camera id.
            - model (object): The loaded model object for a specified camera id.
            - regions_map (dict): The loaded regions map for the specified camera id.
        
        Notes: regions_map is a dictionary that maps zone number to zone letter for each cam_id.
        """

        self.dir_path = Path(dir_path)
        self.save_dir = Path(os.getcwd()) / "data"
        self.location = location
        self.output_file = Path(output_file)
        

        try:
            self.scaler = load(Path(os.getcwd()) / "scalers" / f"{self.location}.pkl")
            logging.info(f"Scaler successfully loaded for {self.location}")
            self.model = load(Path(os.getcwd()) / "models" / self.location / "model.pkl")
            logging.info(f"Model successfully loaded for {self.location}")
        except Exception as e:
            raise CustomException(e, sys)
        
        try:
            with open(Path(os.getcwd()) / "locations" / "regions.json", "r") as f:
                self.regions_map = json.load(f)
            self.regions_map = self.regions_map[location]
        except Exception as e:
            raise CustomException(e, sys)
        

    def make_sequenced_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates a sequenced dataset by generating all possible combinations of zones, classes, and datetime values, 
        and then merging these combinations with the input DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame containing 'datetime', 'zone_in', 'zone_out', and 'class' columns.

        Returns:
            pd.DataFrame: A sequenced DataFrame with 'zone_in', 'zone_out', 'class', 'datetime', 'last_15_min_count', 
            and 'last_30_min_count' columns.
        """

        try:
            class_values = [0,1,2,3,4,5,6]
            datetime_values = df["datetime"].unique()
            zone_in_values = df["zone_in"].unique()
            zone_out_values = df["zone_out"].unique()
            zone_values = set(zone_in_values)
            zone_values.update(zone_out_values)

            all_combinations = list([values for values in itertools.product(zone_values, zone_values, class_values, datetime_values) if values[0] != values[1]])
            all_combinations_df = pd.DataFrame(all_combinations, columns=['zone_in', 'zone_out', 'class','datetime'])
            merged_df = pd.merge(df, all_combinations_df, on=['zone_in', 'zone_out','class','datetime'], how="right")

            merged_df = merged_df.sort_values(by=["datetime","zone_in","zone_out","class"])
            merged_df["count"] = merged_df["count"].fillna(0).astype(int)
            group_val = merged_df.groupby(by=["datetime"], as_index=False).count()["count"].iloc[0]

            merged_df["last_30_min_count"] = merged_df["count"].shift(+group_val)
            merged_df = merged_df.rename(columns={"count":"last_15_min_count"})
        except Exception as e:
            raise CustomException(e, sys)

        return merged_df
    

    def get_dateset_from_dir(self) -> pd.DataFrame:
        """
        Reads extracted counts files from the specified directory and returns a DataFrame with the extracted counts.

        Returns:
            pd.DataFrame: A sequenced DataFrame with 'zone_in', 'zone_out', 'class', 'datetime', 'last_15_min_count', and 'last_30_min_count' columns.
        """
        try:
            dfs = []
            logging.info(f"Reading files from {self.dir_path}")
            for file in os.listdir(self.dir_path):
                if not file.endswith(".csv"):
                    continue
                index = file.removesuffix(".csv").split("_")[-1]
                time_stamp = datetime.datetime.strptime("00:00:00","%H:%M:%S") + datetime.timedelta(minutes=15*int(index))

                df = pd.read_csv(self.dir_path / file)

                if df.empty:
                    print(f"Empty: {file}")
                    continue
                try:
                    df["count"] = df["count"].astype(int)
                except KeyError:
                    df.columns =["time_stamp","zone_in","zone_out","class","count"]

                df = df.loc[(df["class"] != 7)]
                df["datetime"] = time_stamp
                df = df.groupby(["datetime","class","zone_in","zone_out"])[["count"]].sum().reset_index()
                df["datetime"] = pd.to_datetime(df["datetime"])
                dfs.append(df)

            try:
                df = pd.concat(dfs, axis=0, ignore_index=True)
                df = df.sort_values(by=["datetime","zone_in","zone_out","class"])
                df = self.make_sequenced_dataset(df)
                df = df.drop(["datetime"], axis=1)
                df = df.dropna()
                df["class"] = df["class"].astype(np.int8)
                df["zone_in"] = df["zone_in"].astype(np.int8)
                df["zone_out"] = df["zone_out"].astype(np.int8)
                df["last_15_min_count"] = df["last_15_min_count"].astype(np.int16)
                df["last_30_min_count"] = df["last_30_min_count"].astype(np.int16)

            except Exception as e:
                raise CustomException(e, sys)
        except Exception as e:
            raise CustomException(e, sys)
        logging.info(f"Dataset sequenced for {self.location} in {self.dir_path}")

        return df
    
    def get_cumulative_counts(self, df: pd.DataFrame) -> dict:
        """
        Retrieves cumulative counts of vehicles from a given DataFrame.

        Parameters:
            df (pd.DataFrame): A DataFrame containing vehicle data.

        Returns:
            dict: A dictionary of cumulative counts, where each key is a zone pair and each value is another dictionary containing vehicle class counts.
        """

        try:
            df_save = df.copy()
            classes = ["Two Wheeler","Three Wheeler","Car","Bus","LCV","Truck","Bicycle"]  
            df_save["count"] = df_save["last_15_min_count"] + df_save["last_30_min_count"]
            df_save = df_save.drop(["last_15_min_count","last_30_min_count"], axis=1)



            cumulative_counts = defaultdict(lambda : defaultdict(int))
            for _, row in df_save.iterrows():
                zone_pair  = f"{self.regions_map[row['zone_in']]}{self.regions_map[row['zone_out']]}"
                vehicle_class = classes[row["class"]]
                cumulative_counts[zone_pair][vehicle_class] = int(row["count"])

            cumulative_counts = {zone: dict(classes) for zone, classes in cumulative_counts.items()}

        except Exception as e:
            raise CustomException(e, sys)
        
        return cumulative_counts
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Makes predictions for the next 30 minutes of turning movements counts.

        Parameters:
            df (pd.DataFrame): A DataFrame containing the input data.

        Returns:
            pd.DataFrame: A DataFrame containing the predicted turning movements counts for the next 30 minutes.
        """
        
        try:
            df_next = df.copy()
            features = self.scaler.transform(df)
            y_pred_15_min = self.model.predict(features)
            y_pred_15_min = y_pred_15_min.reshape(-1,1)
            y_pred_15_min = np.array(y_pred_15_min, dtype=int)
            
            df_next = df_next.drop(["last_30_min_count"], axis=1)
            df_next = df_next.rename(columns={"last_15_min_count" : "last_30_min_count"})
            df_next["last_15_min_count"] = y_pred_15_min.copy()
            df_next = df_next.iloc[:, [0,1,2,4,3]]

            features = self.scaler.transform(df_next)
            y_pred_30_min = self.model.predict(features)
            y_pred_30_min = y_pred_30_min.reshape(-1,1)
            y_pred_30_min = np.array(y_pred_30_min, dtype=int)

            df_next["last_15_min_count"] = y_pred_15_min
            df_next["last_30_min_count"] = y_pred_30_min

        except Exception as e:
            raise CustomException(e, sys)
        logging.info(f"Turning movements counts for next 30 minutes done.")
        
        return df_next
    
    def get_location(self) -> str:
        return self.location
    
    def get_save_dir(self) -> str:
        return self.save_dir
    
    def get_output_file(self) -> str:
        return self.output_file

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="Traffic Density Predictor"
    )

    parser.add_argument(
        "--dir_path",
        required=True,
        default=f"{Path(os.getcwd()) / 'results' / 'data'}",
        help="Path to extracted counts directory",
        type=str
    )

    parser.add_argument(
        "--location",
        required=True,
        default="SBI_Bnk_JN_FIX_1",
        help="Cam_Id",
        type=str
    )

    parser.add_argument(
        "--output_file",
        required=True,
        default=f"{Path(os.getcwd()) / 'data' / 'output_file.json'}",
        help="Path to output file",
        type=str
    )

    try:
        args = parser.parse_args()

        predictor = CountsPredictor(
            dir_path=args.dir_path,
            location=args.location,
            output_file=args.output_file
        )
        
        dataset = predictor.get_dateset_from_dir()
        cumulative_counts = predictor.get_cumulative_counts(dataset)
        logging.info(f"Cumulative counts extracted.")

        dataset_pred = predictor.predict(dataset)
        predicted_counts = predictor.get_cumulative_counts(dataset_pred)
        logging.info(f"Predicted counts extracted.")

        results = {
            predictor.get_location():{
                "Cumulative Counts": cumulative_counts,
                "Predicted Counts": predicted_counts
            }
        }

        with open(predictor.get_output_file(),"w") as f:
            json.dump(results, f, indent=4)
        
    except Exception as e:
        raise CustomException(e, sys)

    