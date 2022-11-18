from dataclasses import dataclass
from typing import Set


@dataclass
class LaneRoute:
    lane_idxs: Set[int]
