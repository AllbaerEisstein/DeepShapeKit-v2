from dataclasses import dataclass 
from typing import Dict, List

class KeypointsDict(Dict[str, Dict[str, List[float]]]):
    """
    dict[
        str, dict[
    ###  └──> frame number as string
            str, dict[
    ###      └──> instance number 
                str, list[float]
    ###          └──> keypoint name
            ]
        ]
    ]
    """
    pass