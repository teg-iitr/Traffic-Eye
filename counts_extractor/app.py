import os
import sys
import json
import subprocess
from pathlib import Path
from src.logger import logging
from src.expection import CustomException

def main(input_file: str, output_file: str) -> None:

    try:
        with open(input_file, "r") as f:
            data = json.load(f)
        logging.info(f"{input_file} file successfully loaded.")
        
    except Exception as e:
        raise CustomException(e, sys)

    cam_id = list(data.keys())[0]
    command = [
        "python3","-m", "src.pipeline.extractor",
        "--weights_path", "./weights/best_ITD_aug.pt",
        "--location", f"{cam_id}",
        "--input_video", ""
        ]
    
    dir_path = None
    for k,v in data[cam_id].items():
        command[-1] = str(v)

        if dir_path is None:
            dir_path = Path(os.getcwd()) / "results" / Path(v).parent.stem / cam_id
        try:
            result = subprocess.run(command, check=True, text=True)

            logging.info(f"Extraction completed for {k} for {cam_id}.")
        except subprocess.CalledProcessError as e:
            raise CustomException(e, sys)
    print("Extraction done.")
    print("Predicting traffic counts.")
    
    command = [
    "python3","-m", "src.pipeline.predictor",
    "--dir_path", dir_path,
    "--location", f"{cam_id}",
    "--output_file", output_file
    ]

    try:
        result = subprocess.run(command, check=True, text=True)
    except Exception as e:
        raise CustomException(e, sys)
    
    logging.info(f"Outputs saved to {output_file}")
    print(f"Outputs saved to {output_file}")
    


if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    input_file = os.path.join("/app","data", input_file)
    output_file = os.path.join("/app","data", output_file)

    main(
    input_file, 
    output_file
    )
