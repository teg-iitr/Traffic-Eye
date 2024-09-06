import os
import sys
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from ultralytics import YOLO
from typing import List, Tuple
from src.logger import logging
from collections import defaultdict
from src.expection import CustomException
from src.components.models import Track
from src.components.caching import LRUCache
from src.utils import (load_zones, check_in_zones, write_to_csv,
                       write_counts, convert_seconds_to_hms, draw_bouding_box)


class Detections:
    
    def __init__(self) -> None:
        """
        Initializes the object with an LRUCache of capacity 150.
        """
        self.cache = LRUCache(capacity=150)
    
    def update(self, detections:  List[Tuple[int, int, List[float]]]) -> None:
        """
        Updates the cache based on the detections provided.
        
        Parameters:
            detections (List[Tuple[int, int, List[float]]]): A list of detections containing track_id, class_id, and bbox.
        
        Returns:
            None
        """
        for track_id, class_id, bbox in detections:
            track = self.cache.get(track_id)
            if track is None:
                self.cache.put(track_id, Track(track_id, class_id, bbox))
            else:
                track.update(bbox, class_id)
                


class VideoProcessor:
    
    def __init__(self, weights_path: str, input_video: str, device_id: int, location: str = None) -> None:
        """
        Initializes the VideoProcessor object with weights_path, input_video, location, and is_save_vid parameters. 
        Sets up various attributes like weights_path, input_video, location, detection_manager, zones, file_name, save_path, counter, fps, processed_frames, timestamp, and is_save_vid. 
        Tries to load the YOLO model with the specified weights_path, handles exceptions, and logs success message. 
        If location is not provided, extracts it from the file_name. 
        """
        self.weights_path = Path(weights_path)
        self.input_video = Path(input_video)
        self.location = location
        self.detection_manager = Detections()
        self.zones = None
        self.file_name = self.input_video.stem
        self.save_path = Path(os.getcwd()) / "results" / self.input_video.parent.stem
        self.objects_counter = defaultdict(str) # For saving classified counts for each pair on zones
        self.fps = None
        self.processed_frames = 0
        self.timestamp = None
        self.device_id = device_id
        self.zones_counter = defaultdict(lambda: [0]*8) # Counts for vehilce entering zone_out; for annotation purpose
        
        try:
            self.model = YOLO("yolov8n.pt")
            self.model = YOLO(self.weights_path)
            logging.info(f"Model successfully loaded.")
        except Exception as e:
            raise CustomException(e, sys)
        
        try:
            if not self.location:
                # If location is not provided, extract location from file_name
                self.location = self.file_name.split("_vid_")[0]
        except Exception as e:
            raise CustomException(e, sys)        

    def process_video(self, is_render: bool, is_save_vid: bool, outfps: int) -> None:
        """
        Process a video by loading it, processing each frame, and saving the results.
        
        Parameters:
            is_render (bool): Whether to render the annotated frames.
            is_save_vid (bool): Whether to save the annotated video.
            outfps (int): The output frames per second. If -1, the original fps is used.
        
        Returns:
            None
        """
        
        try:
            self.save_path = self.save_path / self.location
            os.makedirs(self.save_path, exist_ok=True)
            
            cap = cv2.VideoCapture(self.input_video)
            self.fps = cap.get(cv2.CAP_PROP_FPS)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Load zones for the location
            self.zones = load_zones(self.location, self.input_video)
            
            if self.zones is None: 
                raise CustomException("No zones found. Please provide the correct location",sys)
            
            
            # Initialize csv file
            row = ["time_stamp","zone_in","zone_out","class","count"]
            write_to_csv(self.save_path / f"{self.file_name}.csv", row, mode="w")
            
            
            if outfps <= -1:
                skip_frames = 1
            else:
                skip_frames = outfps

            with tqdm(total=total_frames, desc=f"Processing video: {self.file_name}") as pbar:
                logging.info("Video processing started for {}, fps: {}, frame_width: {}, frame_height: {}, total_frames: {}".format(
                    self.input_video.name,
                    self.fps,
                    frame_width,
                    frame_height,
                    total_frames
                ))
                
                if is_save_vid:
                    out = cv2.VideoWriter(self.save_path / f"{self.file_name}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), self.fps//skip_frames, (frame_width, frame_height))
                
                skipped_frames = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    
                    self.processed_frames += 1

                    if skip_frames != 1:
                        skipped_frames += 1

                    if not ret:
                        break

                    if skip_frames != 1 and skipped_frames < skip_frames:
                        pbar.update(1)
                        continue

                    if skip_frames != 1:
                        skipped_frames = 0
                    
                    annotated_frame = self.process_frame(frame)
                    
                    if is_save_vid:
                        out.write(annotated_frame)
                    
                    if is_render:
                        resized_frame = cv2.resize(annotated_frame, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                        cv2.imshow("Annotated Frame", resized_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    
                    pbar.update(1)

            
        except Exception as e:
            raise CustomException(e, sys) 
        finally:
            #  save counts for the complete video
            if len(self.objects_counter) != 0:
                timestamp = convert_seconds_to_hms(self.processed_frames/self.fps)
                write_counts(self.save_path / f"{self.file_name}.csv", self.objects_counter, timestamp)
                logging.info("Counts saved for {} at {}".format(self.input_video.name, timestamp))
                
            logging.info("Data extration completed for {}, results saved in {}".format(self.input_video.name, self.save_path))
            if is_save_vid:
                out.release()
            cap.release()
            cv2.destroyAllWindows()


    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame from a video stream.

        Parameters:
        frame (np.ndarray): The input frame to be processed.

        Returns:
        np.ndarray: The annotated frame with detected objects.
        """
        
        try:

            results = list(self.model.track(frame, persist=True, verbose=False, device=self.device_id, imgsz=992 ,tracker="bytetrack.yaml"))[0]

            
            bbox_xyxys = results.boxes.xyxy.cpu().numpy().tolist()
            bbox_confs = results.boxes.conf.cpu().numpy().tolist()
            class_ids = results.boxes.cls.cpu().numpy().tolist()
            
            # If no detections, initailize tracks_ids with empty list
            if results.boxes.id is None:
                tracks_ids =  []
            else:
                tracks_ids = results.boxes.id.cpu().numpy().tolist()
            
            detections = list(zip(tracks_ids, class_ids, bbox_xyxys))
            
            # Update cache
            self.detection_manager.update(detections)
            
            self.detections_in_zones(tracks_ids)
            
            annotated_frame = self.annotate_frame(frame, tracks_ids)
            
            # Removed; save counts for the complete video
            
            # seconds = int(self.processed_frames/self.fps)
            # if seconds != 0 and seconds%300 == 0:
            #     # save counts every 5 minutes
            #     timestamp = convert_seconds_to_hms(seconds)
            #     if self.timestamp != timestamp:
            #         write_counts(self.save_path / f"{self.file_name}.csv", self.objects_counter, timestamp)
            #         self.objects_counter = defaultdict(str)
            #         logging.info("Counts saved for {} at {}".format(self.input_video.name, timestamp))
            #         self.timestamp = timestamp
                        
            return annotated_frame
        except Exception as e:
            raise CustomException(e, sys)
    
    
    def annotate_frame(self, frame: np.ndarray, detections: List[float]) -> np.ndarray:
        annotated_frame = frame.copy()
        
        classes = ["MTW","TRW","Car","Bus","LCV","Truck","Cycle","Person"]
        colors = [(0, 128, 255), (0, 200, 128), (0, 0, 255), (255, 255, 0), (127, 0, 255), (204, 0, 0), (0,255,255), (153, 0, 153)]
        
        try:
            # Draw zones
            dy = 35
            for i,zone in enumerate(self.zones):
                zone_label = "Zone {}".format(i)
                t_size = cv2.getTextSize(zone_label, 0, fontScale=1, thickness=2)[0]
                c2 = zone[0][0] + t_size[0] + 2, zone[0][1] + t_size[1] + 5
                cv2.rectangle(annotated_frame, zone[0], c2, (245,135,66), -1, cv2.LINE_AA)
                cv2.putText(annotated_frame, zone_label, (zone[0][0], c2[1] - 5), 0, 1, (0,0,0), thickness=2, lineType=cv2.LINE_AA)
                cv2.polylines(annotated_frame, [np.array(zone)], isClosed=True, color=(245,135,66), thickness=2)
                
                # Drow counters for each zone
                anchor = (int((zone[0][0]+zone[1][0])/2), zone[0][1])
                for j, cls_cnt in enumerate(self.zones_counter[i]):
                    cls_label = "{}:{}".format(classes[j], cls_cnt)
                    t_size = cv2.getTextSize(cls_label, 0, fontScale=1, thickness=2)[0]
                    c2 = anchor[0] + t_size[0] + 2, anchor[1] + t_size[1] + 5
                    cv2.rectangle(annotated_frame, (anchor[0], anchor[1]+ j*dy), (c2[0], c2[1] + j*dy), (82,168,50), -1, cv2.LINE_AA)
                    cv2.putText(annotated_frame, cls_label, (anchor[0], c2[1] + j*dy), 0, 1, (0,0,0), thickness=2, lineType=cv2.LINE_AA)
                    
                    
            for track_id in detections:
                track_id = int(track_id)
                track = self.detection_manager.cache.get(track_id)
                if track is not None:
                    cx, cy = track.get_centroid()
                    class_id = track.get_track_cls()
                    x1,y1,x2,y2 = track.get_track_bbox()
                    
                    
                    label = "{}:{}".format(track_id, classes[class_id])
                    annotated_frame = draw_bouding_box(annotated_frame, (x1,y1), (x2,y2), colors[class_id], label)
                    cv2.circle(annotated_frame, (cx,cy), 5,  colors[class_id], cv2.FILLED)
                    
                    trails = track.get_trail()
                    for i in range(1,len(trails)):
                        cv2.line(annotated_frame, (trails[i][0],trails[i][1]), (trails[i-1][0],trails[i-1][1]), colors[class_id], thickness=2)
                        
            return annotated_frame
        
        except Exception as e:
            raise CustomException(e, sys)
                
                        
        
    def detections_in_zones(self, detections: List[float]):
        """
        Updates the zones of detected objects based on their current positions.

        Args:
            detections (List[float]): A list of track IDs of detected objects.

        Returns:
            None
            
        Notes:
            zone_in is upadated once, zone_out is updated if the track enters a new zone other than zone_in and current zone_out.
        """
        
        try:

            for track_id in detections:
                track_id = int(track_id)
                track = self.detection_manager.cache.get(track_id)
                if track is not None:
                    cx,cy = track.get_centroid()
                    zone_in, zone_out = track.get_zones()
                    class_id = track.get_track_cls()
                        
                    zone_ind = check_in_zones(self.zones, (cx,cy))

                    if zone_ind != -1:
                        if zone_in is None:
                            zone_in = zone_ind
                        elif zone_out != zone_in and zone_ind != zone_in and zone_out != zone_ind:
                            zone_out = zone_ind
                            
                            key = f"{track_id},{str(class_id)}"
                            
                            self.objects_counter[key] = f"{zone_in},{zone_out}" 
                            
                            self.zones_counter[zone_out][class_id] += 1
                            
                        track.set_zones((zone_in,zone_out))
        except Exception as e:
            raise CustomException(e, sys)            
                    
                    
        
        


if __name__ == "__main__":
    
    # if torch.cuda.is_available():
    #     torch.cuda.set_device(0)
    # else:
    #     torch.cuda.set_device("cpu")
    #     logging.info("CUDA is not available. Running on CPU.")

    
    parser = argparse.ArgumentParser(
        description="Traffic Density Extractor"
    )
    
    parser.add_argument(
        "--weights_path",
        required=True,
        default="path/to/best.pt",
        help="Path to the weights file",
        type=str
    )
    
    parser.add_argument(
        "--input_video",
        required=True,
        help="Path to the input video file",
        type=str
    )
    
    parser.add_argument(
        "--location",
        help="Name of the location for ROI initailization",
        type=str
    )
    
    parser.add_argument(
        "--is_save_vid",
        help="Whether to save the annotated video or not",
        type=bool,
        default=False
    )
    
    parser.add_argument(
        "--device_id",
        default=0,
        type=int,
        help="GPU device ID for inference; or pass cpu if GPU not available"
    )
    
    parser.add_argument(
        "--is_render",
        default=False,
        type=bool,
        help="Whether to render the video or not"
    )
    parser.add_argument(
        "--outfps",
        default=-1,
        type=int,
        help="Rate of detection; outfps/fps = your desired fps \
            Ex: fps = 25; desired fps = 5 or 0.2 \
                Then outfps = 5 \
                Fallback to fps if outfps is not specified \
                Note: more outfps value leads to faster processing but less accuracy; outfps value less than 5 recommended"
    )
    

    args = parser.parse_args()
    
    processor = VideoProcessor(
        weights_path=args.weights_path,
        input_video=args.input_video,
        device_id=args.device_id,
        location=args.location,
    )
    
    processor.process_video(
        is_render=args.is_render,
        is_save_vid=args.is_save_vid,
        outfps=args.outfps
        )
    
    
     
