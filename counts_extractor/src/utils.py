import os
import cv2
import csv
import sys
import yaml
import numpy as np
from pathlib import Path
from typing import List, Tuple
from src.logger import logging
from collections import defaultdict
from src.expection import CustomException

def draw_bouding_box(image: np.ndarray, top_left: Tuple[int, int], bottom_right: Tuple[int, int], color: Tuple[int,int,int], label: str, roundness: float = 0.6) -> np.ndarray:
    """
    Draws a bounding box with rounded corners around a track.

    Parameters:
    image (np.ndarray): The input image.
    top_left (Tuple[int, int]): The coordinates of the top-left corner of the bounding box.
    bottom_right (Tuple[int, int]): The coordinates of the bottom-right corner of the bounding box.
    color (Tuple[int, int, int]): The color of the bounding box.
    label (str): The label to be displayed next to the bounding box.
    roundness (float, optional): The roundness of the corners of the bounding box. Defaults to 0.6.

    Returns:
    np.ndarray: The annotated image with the bounding box.
    """
    
    annotated_image = image.copy()

    # radius of the rounded corners
    radius = (
        int((bottom_right[0]-top_left[0]) // 2*roundness)
        if abs(top_left[0] - bottom_right[0]) < abs(top_left[1] - bottom_right[1])
        else int((bottom_right[1]-top_left[1]) // 2*roundness)
    )    

    
    p1 = top_left
    p2 = (bottom_right[0], top_left[1])
    p3 = bottom_right
    p4 = (top_left[0], bottom_right[1])
    
    cv2.line(annotated_image, (p1[0] + radius,p1[1]), (p2[0] - radius,p2[1]), color=color, thickness=2, lineType=cv2.LINE_AA)
    cv2.line(annotated_image, (p2[0],p2[1] + radius), (p3[0],p3[1] - radius), color=color, thickness=2, lineType=cv2.LINE_AA)
    cv2.line(annotated_image, (p4[0] + radius, p4[1]), (p3[0] - radius, p3[1]), color=color, thickness=2, lineType=cv2.LINE_AA)
    cv2.line(annotated_image, (p1[0],p1[1] + radius), (p4[0], p4[1] - radius), color=color, thickness=2, lineType=cv2.LINE_AA)
    
    # rounded corners
    cv2.ellipse(annotated_image, (p1[0] + radius, p1[1] + radius), (radius, radius), 180, 0, 90, color=color, thickness=2, lineType=cv2.LINE_AA)
    cv2.ellipse(annotated_image, (p2[0] - radius, p2[1] + radius), (radius, radius), 270, 0, 90, color=color, thickness=2, lineType=cv2.LINE_AA)
    cv2.ellipse(annotated_image, (p3[0] - radius, p3[1] - radius), (radius, radius), 0, 0, 90, color=color, thickness=2, lineType=cv2.LINE_AA)
    cv2.ellipse(annotated_image, (p4[0] + radius, p4[1] - radius), (radius, radius), 90, 0, 90, color=color, thickness=2, lineType=cv2.LINE_AA)
    
    # text label
    t_size = cv2.getTextSize(label, 0, fontScale=0.7, thickness=2)[0]
    c2 = p1[0] + radius + t_size[0] + 2, p1[1] + t_size[1] + 5
    cv2.rectangle(annotated_image, (p1[0] + radius, p1[1]), c2, color, -1, cv2.LINE_AA)
    cv2.putText(annotated_image, label, (p1[0] + radius, c2[1] - 5), 0, 0.7, (0,0,0), thickness=1, lineType=cv2.LINE_AA)
        
    return annotated_image


def capture_points(image: np.ndarray) -> List[list]:
    """
    Captures points from an image by displaying the image and waiting for left mouse button clicks.
    
    Parameters:
    image (np.ndarray): The input image to capture points from.
    
    Returns:
    List[list]: A list of captured points, where each point is a list of two integers representing the x and y coordinates.
    """
    try:
        roi = []

        def click_event(event, x, y, flags, param):
            """
            Handles the click event for the image.
            This function is called when a click event occurs on the image. It checks if the event is a left mouse button click (cv2.EVENT_LBUTTONDOWN).
            If it is, it appends the scaled coordinates of the click event to the 'roi' list and prints the captured point coordinates. 
            It then draws a green circle at the clicked position on the image. Finally, it updates the display window with the image.

            Note:
                This function assumes that the 'image' variable is defined and accessible within the scope of this function.
            """
            
            if event == cv2.EVENT_LBUTTONDOWN:
                roi.append([int(2*x),int(2*y)])
                print(f"Point captured: ({x}, {y})")

                cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

            cv2.imshow('Image', image)


        cv2.namedWindow('Image')
        cv2.setMouseCallback('Image', click_event)
        print("Click on the image to capture points:")

        cv2.imshow('Image', image)
        cv2.waitKey(0)


        return roi
    except Exception as e:
        raise CustomException(e, sys)
    finally:
        cv2.destroyAllWindows()

def draw_zones(video_path: str) -> List[list]:
    """
    Draws zones on a video frame based on user input.

    Args:
        video_path (str): The path to the video file.

    Returns:
        List[list]: A list of zones, where each zone is a list of points.

    Notes:
        This function uses OpenCV to read the video frame and display it to the user.
        The user is prompted to select at least three points for each zone.
        The function continues to prompt the user for additional zones until they choose to stop.
    """
    x0, y0 = 100, 100
    
    try:
        cap = cv2.VideoCapture(video_path)
        _, image = cap.read()
        
        text = "Look at terminal for directions"
        t_size = cv2.getTextSize(text, 0, fontScale=1, thickness=2)[0]
        c2 = x0 + t_size[0] + 2, y0 - t_size[1] - 5
        
        cv2.rectangle(image, (x0,y0), c2, (255,0,255), -1, cv2.LINE_AA)
        cv2.putText(image, text, (x0,y0), 0, 1, (10,10,10), 1, lineType=cv2.LINE_AA)
        image = cv2.resize(image, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        
        
        zones = []
        while not zones or len(zones[0]) < 3:
            zones = []
            print("Choose at least 3 points for zone, press any key once done.")
            roi = capture_points(image)
            zones.append(roi[:])
            
        while (respose := input("Do you want to continue? (y/n)")) != "n":
            for zone in zones:
                cv2.polylines(image, [np.array(zone, dtype=int)//2], True, (255, 255, 0), 2)
            
            roi = capture_points(image)
            zones.append(roi[:])
            
        return zones.copy()
            
    except Exception as e:
        raise CustomException(e, sys)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
def load_zones(location: str, video_path: str) -> List[list]:
    """
    Loads zone data for a specific location from a YAML file. 
    If the location does not exist, another function is called to draw zones on the fly and save them.

    Args:
        location (str): The location for which to load zone data.
        video_path (str): The path to a video file used to draw zones if they do not exist.

    Returns:
        List[list]: A list of zone data, where each zone is a list of points.
    """
    try:
        data_path = Path(os.getcwd()) / "locations" / "locations.yaml"
        
        with open(data_path, "r") as file:
            data = yaml.safe_load(file)
            
        if data is None or data.get(location) is None:
            logging.info("No zones found for {}.".format(location))
            if data is None: 
                data = dict()
            zones_data = draw_zones(video_path)
            
            with open(data_path, "w") as file:
                data[location] = zones_data
                yaml.dump(data, file)
                logging.info("Zones created for {} in {}".format(location, data_path))
                
        data = data.get(location)
        
        return data 
    
    except Exception as e:
        raise CustomException(e, sys)
    
    
def check_in_zone(zone: List[Tuple[int, int]], point: Tuple[int, int]) -> bool:
    return cv2.pointPolygonTest(np.array(zone, dtype=int), point, False) >= 0 


def check_in_zones(zones: List[List[Tuple[int, int]]], point: Tuple[int, int]) -> bool:
    for i,zone in enumerate(zones):
        if check_in_zone(zone, point):
            return i
    return -1


def write_to_csv(file_path: str, data: List[str], mode="a") -> None:
    with open(file_path, mode, newline="") as file:
        writer = csv.writer(file)
        writer.writerow(data)
        
        
def convert_seconds_to_hms(seconds: int) -> str:
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

def write_counts(file_path: str, data: dict, timestamp: str) -> None:
    """
    Writes counts data to a CSV file.

    Args:
        file_path (str): The path to the CSV file.
        data (dict): A dictionary containing the counts data.
        timestamp (str): The timestamp to be written to the CSV file.

    Returns:
        None
    """
    counter = defaultdict(str)
    for key, zones in data.items():
        _, cls = key.split(",")
        cnt_key = zones
        
        if cnt_key not in counter:
            counter[cnt_key] = defaultdict(int)
        counter[cnt_key][cls] += 1
        
    for key in counter:
        zone_in, zone_out = key.split(",")
        for class_id, count in counter[key].items():
            row = [timestamp,zone_in, zone_out, class_id, count]
            write_to_csv(file_path, row)
    
    
    