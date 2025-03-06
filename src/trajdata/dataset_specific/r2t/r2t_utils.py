import os
from typing import Optional

from trajdata.data_structures.agent import AgentMetadata, AgentType, FixedExtent


R2T_DATASET_NAME = "r2t"
R2T_DT = 0.1  # 3d annotations, images
R2T_SPLITS = ("train", "val", "test")

NUSCENES_CLASS_MAPPING = {
    'car': 0,
    'truck': 1,
    # 'construction_vehicle': 2,
    'bus': 3,
    # 'trailer': 4,
    'other vehicle': 4,
    # 'barrier': 5,
    'motorcyclist': 6,
    'cyclist': 7,
    'pedestrian': 8,
    # 'traffic_cone': 9,
    'animals': 10,
}

R2T_SCENARIO_OBS_TIMESTEPS = 4  # frames, @ 2 Hz
R2T_SCENARIO_TOTAL_TIMESTEPS = 10

def nusc_type_to_unified_type(nusc_type: int) -> AgentType:
    if nusc_type == 0:
        return AgentType.VEHICLE
    elif nusc_type == 1:
        return AgentType.VEHICLE
    elif nusc_type == 2:
        return AgentType.VEHICLE
    elif nusc_type == 3:
        return AgentType.VEHICLE
    elif nusc_type == 4:
        return AgentType.VEHICLE
    elif nusc_type == 5:
        return AgentType.UNKNOWN
    elif nusc_type == 6:
        return AgentType.MOTORCYCLE
    elif nusc_type == 7:
        return AgentType.BICYCLE
    elif nusc_type == 8:
        return AgentType.PEDESTRIAN
    else:
        return AgentType.UNKNOWN

