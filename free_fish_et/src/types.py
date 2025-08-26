from dataclasses import dataclass 
from typing import Dict, List

class KeypointsDict(Dict[str, Dict[str, List[float]]]):
    """
    A nested dictionary structure for storing keypoints data.

    Structure:
        KeypointsDict[instance_number][keypoint_name] -> List[float]

    - instance_number (str): The instance identifier within the frame.
    - keypoint_name (str): The name of the keypoint.
    - List[float]: The coordinates or values associated with the keypoint.

    Example:
    {
        "0": {
            "nose": [x, y, confidence],
            "tail": [x, y, confidence]
        },
        "1": {
            "nose": [x, y, confidence],
            "tail": [x, y, confidence]
        },
    }
    """
    pass