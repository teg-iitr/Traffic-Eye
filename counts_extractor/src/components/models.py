from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Union

@dataclass
class Track:
    def __init__(self, track_id: int, track_cls: float, track_bbox: List[int]) -> None:
        """
        Initializes the Track object with the provided track_id, track_cls, and track_bbox.
        zone_in and zone_out attributes are set to None by default, which are used to determining the turning movement of a track.
        
        Parameters:
            track_id (int): The ID of the track.
            track_cls (str): The class of the track.
            track_bbox (List[int]): The bounding box coordinates of the track.
        
        Returns:
            None
        """
        self.track_id = int(track_id)
        self.track_cls = int(track_cls)
        self.track_bbox = [int(val) for val in track_bbox]
        self.centroid = self.calculate_centroid()
        self.trail = deque(maxlen=3)
        self.trail.append(self.centroid)

        self.zone_in = None
        self.zone_out = None

    
    def update(self, track_bbox: List[int], track_cls: float) -> None:
        """
        Updates the track's bounding box and class, recalculates its centroid, and appends the new centroid to the trail.

        Parameters:
            track_bbox (List[int]): The new bounding box coordinates of the track.
            track_cls (float): The new class of the track.

        Returns:
            None
        """
        
        self.track_bbox = [int(val) for val in track_bbox]
        self.track_cls = int(track_cls)
        self.centroid = self.calculate_centroid()
        self.trail.append(self.centroid)
        
    def calculate_centroid(self) -> Tuple[int, int]:
        """
        Calculates the centroid of the track's bounding box.

        Parameters:
            None

        Returns:
            Tuple[int, int]: The x and y coordinates of the centroid.
        """
        cx = int((self.track_bbox[0]+self.track_bbox[2])/2)
        cy = int((self.track_bbox[1]+self.track_bbox[3])/2)
        # cy = int(self.track_bbox[3])
        return (cx,cy)
    
    def set_zones(self, zones: List[int]) -> None:
        """
        Sets the zone_in and zone_out attributes; used for counting.

        Parameters:
            zones (List[int]): A list containing the zone_in and zone_out values.

        Returns:
            None
        """
        self.zone_in, self.zone_out = zones
        
    def get_track_id(self) -> int:
        return self.track_id
    
    def get_track_cls(self) -> int:
        return self.track_cls
    
    def get_track_bbox(self) -> List[int]:
        return self.track_bbox
    
    def get_trail(self) -> List[Tuple[int, int]]:
        return self.trail
        
    def get_centroid(self) -> Tuple[int, int]:
        return self.centroid
    
    def get_zones(self) -> Union[Tuple[None,None],Tuple[int, int]]:
        return (self.zone_in, self.zone_out)
    
    