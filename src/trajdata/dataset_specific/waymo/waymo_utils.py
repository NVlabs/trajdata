import copy
import pathlib
import time
from multiprocessing import Pool
from typing import Dict, Final, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from waymo_open_dataset.protos import map_pb2 as waymo_map_pb2
from waymo_open_dataset.protos import scenario_pb2

from trajdata.maps import TrafficLightStatus
from trajdata.proto import vectorized_map_pb2

WAYMO_DT: Final[float] = 0.1
WAYMO_DATASET_NAMES = ["testing",
                 "testing_interactive",
                 "training",
                 "training_20s",
                 "validation",
                 "validation_interactive"]
from trajdata.data_structures.agent import (
    Agent,
    AgentMetadata,
    AgentType,
    FixedExtent,
    VariableExtent,
)


def parse_data(data):
    scenario = scenario_pb2.Scenario()
    scenario.ParseFromString(data)
    return scenario
class WaymoScenarios:
    def __init__(self, dataset_name, source_dir, load=True, num_parallel_reads=None):
        if dataset_name not in WAYMO_DATASET_NAMES:
            raise RuntimeError('Wrong dataset name. Please choose name from '+str(WAYMO_DATASET_NAMES))
        self.name = dataset_name
        self.source_dir = source_dir
        self.scenarios = []
        if load:
            self.load_scenarios(num_parallel_reads)
    def load_scenarios(self, num_parallel_reads, verbose=True):
        self.scenarios = []
        source_it = pathlib.Path().glob(self.source_dir+'/'+self.name + "/*.tfrecord")
        file_names = [str(file_name) for file_name in source_it if file_name.is_file()]
        if verbose:
            print("Loading tfrecord files...")
        dataset = tf.data.TFRecordDataset(file_names, compression_type='', num_parallel_reads=num_parallel_reads).as_numpy_iterator()

        if verbose:
            print("Converting to protobufs...")
        start = time.perf_counter()
        dataset = np.fromiter(dataset, bytearray)
        # use multiprocessing:
        # self.scenarios = Pool().map(parse_data, dataset)
        # use np vectorization (faster with my computer):
        parser = np.vectorize(parse_data)
        self.scenarios = parser(dataset)
        print(time.perf_counter()-start)
        if verbose:
            print(str(len(self.scenarios)) + " scenarios from " + str(len(file_names)) + " file(s) have been loaded successfully")


def extract_vectorized(map_features: List[waymo_map_pb2.MapFeature], map_name) -> vectorized_map_pb2.VectorizedMap:
    vec_map = vectorized_map_pb2.VectorizedMap()
    vec_map.name = map_name
    shifted_origin = vectorized_map_pb2.Point()
    shifted_origin.x = 0.0
    shifted_origin.y = 0.0
    shifted_origin.z = 0.0
    vec_map.shifted_origin.CopyFrom(shifted_origin)
    max_pt = vectorized_map_pb2.Point()
    max_pt.x = 0.0
    max_pt.y = 0.0
    max_pt.z = 0.0
    min_pt = vectorized_map_pb2.Point()
    min_pt.x = 0.0
    min_pt.y = 0.0
    min_pt.z = 0.0

    for map_feature in tqdm(map_features, desc="Converting the waymo map features into vector map"):
        new_element: vectorized_map_pb2.MapElement = vec_map.elements.add()
        new_element.id = map_feature.id
        if map_feature.HasField("lane"):
            road_lane, temp_max_pt, temp_min_pt = translate_lane(map_feature.lane)
            new_element.road_lane.CopyFrom(road_lane)
        elif map_feature.HasField("crosswalk"):
            ped_crosswalk, temp_max_pt, temp_min_pt = translate_crosswalk(map_feature.crosswalk)
            new_element.ped_crosswalk.CopyFrom(ped_crosswalk)
        else:
            continue
        max_pt.x, max_pt.y, max_pt.z = get_larger_elems([max_pt.x, max_pt.y, max_pt.z], temp_max_pt)
        min_pt.x, min_pt.y, min_pt.z = get_smaller_elems([min_pt.x, min_pt.y, min_pt.z], temp_min_pt)
    vec_map.max_pt.CopyFrom(max_pt)
    vec_map.min_pt.CopyFrom(min_pt)
    return vec_map

def translate_agent_type(agent_type):
    if agent_type == scenario_pb2.Track.ObjectType.TYPE_VEHICLE:
        return AgentType.VEHICLE
    elif agent_type == scenario_pb2.Track.ObjectType.TYPE_PEDESTRIAN:
        return AgentType.PEDESTRIAN
    elif agent_type == scenario_pb2.Track.ObjectType.TYPE_CYCLIST:
        return AgentType.BICYCLE
    elif agent_type == scenario_pb2.Track.ObjectType.OTHER:
        return AgentType.UNKNOWN
    return -1

def get_larger_elems(list1, list2):
    if len(list1) != len(list2):
        return -1
    return [np.max([list1[i], list2[i]]) for i in range(len(list1))]

def get_smaller_elems(list1, list2):
    if len(list1) != len(list2):
        return -1
    return [np.min([list1[i], list2[i]]) for i in range(len(list1))]

def translate_poly_line(polyline: List[waymo_map_pb2.MapPoint]) -> (vectorized_map_pb2.Polyline, List[int, int, int], List[int, int, int]):
    ret = vectorized_map_pb2.Polyline()
    max_pt = [0.0, 0.0, 0.0]
    min_pt = [0.0, 0.0, 0.0]
    for point in polyline:
        ret.dx_mm.add(round(point.x * 100))
        ret.dy_mm.add(round(point.y * 100))
        ret.dz_mm.add(round(point.z * 100))
        max_pt = get_larger_elems(max_pt, [point.x, point.y, point.z])
        min_pt = get_smaller_elems(min_pt, [point.x, point.y, point.z])
    return ret, max_pt, min_pt


def translate_lane(lane: waymo_map_pb2.LaneCenter) -> (vectorized_map_pb2.RoadLane, List[int, int, int], List[int, int, int]):
    road_lane = vectorized_map_pb2.RoadLane
    # road_area = vectorized_map_pb2.RoadArea
    # Waymo lane only gives boundary indices and no center indices
    indices = [lane.left_boundaries.lane_start_index,
               lane.left_bounaries.lane_end_index,
               lane.right_boundaries.lane_start_index,
               lane.right_bounaries.lane_end_index]
    indices.sort()
    left_boundary: List[waymo_map_pb2.MapPoint] = \
        lane.polyline[lane.left_boundaries.lane_start_index:lane.left_bounaries.lane_end_index]
    right_boundary: List[waymo_map_pb2.MapPoint] = \
        lane.polyline[lane.right_boundaries.lane_start_index:lane.right_bounaries.lane_end_index]
    center = lane.polyline[:indices[0]] + lane.polyline[indices[1]:indices[2]] + lane.polyline[indices[3]:]
    road_lane.center, center_max, center_min = translate_poly_line(center)
    road_lane.left_boundary, left_max, left_min = translate_poly_line(left_boundary)
    road_lane.right_boundary, right_max, right_min = translate_poly_line(right_boundary)

    road_lane.entry_lanes.extent(lane.entry_lanes)
    road_lane.exit_lanes.extent(lane.exit_lanes)
    road_lane.adjacent_lanes_left.extent([neighbor.feature_id for neighbor in lane.left_neighbors])
    road_lane.adjacent_lanes_right.extent([neighbor.feature_id for neighbor in lane.right_neighbors])
    max_point = [ np.max([center_max[i], left_max[i], right_max[i]]) for i in range(3)]
    min_point = [ np.min([center_min[i], left_min[i], right_min[i]]) for i in range(3)]
    return road_lane, max_point, min_point


def translate_crosswalk(lane: waymo_map_pb2.Crosswalk) -> (vectorized_map_pb2.PedCrosswalk, List[int, int, int], List[int, int, int]):
    ret = vectorized_map_pb2.PedCrosswalk()
    ret.polygon, max_pt, min_pt = translate_poly_line(lane.polygon)
    return ret, max_pt, min_pt

def extract_traffic_lights(dynamic_states: List[scenario_pb2.DynamicMapState]) -> Dict[Tuple[int, int], TrafficLightStatus]:
    ret: Dict[Tuple[int, int], TrafficLightStatus] = {}
    for i, lane_states in enumerate(dynamic_states):
        for lane_state in lane_states:
            ret[(lane_state.lane, i)] = translate_traffic_state(lane_state.state)
    return ret

def translate_traffic_state(state: waymo_map_pb2.TrafficSignalLaneState.State) -> TrafficLightStatus:
    # The traffic light type doesn't align between waymo and trajdata,
    # but I think trajdata TrafficLightStatus should include yellow light
    # for now I let caution = red
    green = [waymo_map_pb2.TrafficSignalLaneState.State.LANE_STATE_ARROW_GO,
             waymo_map_pb2.TrafficSignalLaneState.State.LANE_STATE_GO,
             ]
    red = [waymo_map_pb2.TrafficSignalLaneState.State.LANE_STATE_ARROW_CAUTION,
           waymo_map_pb2.TrafficSignalLaneState.State.LANE_STATE_ARROW_STOP,
           waymo_map_pb2.TrafficSignalLaneState.State.LANE_STATE_STOP,
           waymo_map_pb2.TrafficSignalLaneState.State.LANE_STATE_CAUTION,
           waymo_map_pb2.TrafficSignalLaneState.State.LANE_STATE_FLASHING_STOP,
           waymo_map_pb2.TrafficSignalLaneState.State.LANE_STATE_FLASHING_CAUTION,
           ]
    if state in green:
        return TrafficLightStatus.GREEN
    if state in red:
        return TrafficLightStatus.RED
    return TrafficLightStatus.UNKNOWN
