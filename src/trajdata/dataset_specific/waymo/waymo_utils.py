import os.path
import pathlib
from typing import Dict, Final, List, Tuple
from subprocess import check_call, check_output
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

TRAIN_SCENE_LENGTH = 91
VAL_SCENE_LENGTH = 91
TEST_SCENE_LENGTH = 11
TRAIN_20S_SCENE_LENGTH = 201

GREEN = [waymo_map_pb2.TrafficSignalLaneState.State.LANE_STATE_ARROW_GO,
         waymo_map_pb2.TrafficSignalLaneState.State.LANE_STATE_GO,
         ]
RED = [waymo_map_pb2.TrafficSignalLaneState.State.LANE_STATE_ARROW_CAUTION,
       waymo_map_pb2.TrafficSignalLaneState.State.LANE_STATE_ARROW_STOP,
       waymo_map_pb2.TrafficSignalLaneState.State.LANE_STATE_STOP,
       waymo_map_pb2.TrafficSignalLaneState.State.LANE_STATE_CAUTION,
       waymo_map_pb2.TrafficSignalLaneState.State.LANE_STATE_FLASHING_STOP,
       waymo_map_pb2.TrafficSignalLaneState.State.LANE_STATE_FLASHING_CAUTION,
       ]

from trajdata.data_structures.agent import (
    Agent,
    AgentMetadata,
    AgentType,
    FixedExtent,
    VariableExtent,
)


class WaymoScenarios:
    def __init__(self, dataset_name, source_dir, download=False, split=False):
        if dataset_name not in WAYMO_DATASET_NAMES:
            raise RuntimeError('Wrong dataset name. Please choose name from ' + str(WAYMO_DATASET_NAMES))
        self.name = dataset_name
        self.source_dir = source_dir
        self.split = split
        if dataset_name in ["training"]:
            self.scene_length = TRAIN_SCENE_LENGTH
        elif dataset_name in ["validation", "validation_interactive"]:
            self.scene_length = VAL_SCENE_LENGTH
        elif dataset_name in ["testing", "testing_interactive"]:
            self.scene_length = TEST_SCENE_LENGTH
        elif dataset_name in ["training_20s"]:
            self.scene_length = TRAIN_20S_SCENE_LENGTH
        if download:
            self.download_dataset()
        if split:
            self.split_scenarios()
        else:
            self.num_scenarios = len(os.listdir(os.path.join(
                self.source_dir, self.name+"_splitted")))

    def download_dataset(self):
        # check_call("snap install google-cloud-sdk --classic".split())
        gsutil = check_output(["which", "gsutil"])
        download_cmd = (str(gsutil.decode("utf-8"))+"-m cp -r gs://waymo_open_dataset_motion_v_1_1_0/uncompressed/scenario/"+str(self.name)+" "+str(self.source_dir)).split()
        check_call(download_cmd)

    def split_scenarios(self, num_parallel_reads=20, verbose=True):
        source_it = pathlib.Path(self.source_dir/self.name).glob("*")
        file_names = [str(file_name) for file_name in source_it]
        if verbose:
            print("Loading tfrecord files...")
        dataset = tf.data.TFRecordDataset(file_names, compression_type='', num_parallel_reads=num_parallel_reads)

        if verbose:
            print("Splitting tfrecords...")

        splitted_dir = os.path.join(self.source_dir, self.name+"_splitted")
        if not os.path.exists(splitted_dir):
            os.makedirs(splitted_dir)
        i = 0
        for data in tqdm(dataset):
            file_name = os.path.join(splitted_dir, self.name+"_splitted_"+str(i)+ ".tfrecords")
            with tf.io.TFRecordWriter(file_name) as file_writer:
                file_writer.write(data.numpy())
            i += 1
        self.num_scenarios = i
        if verbose:
            print(str(i) + " scenarios from " + str(len(file_names)) + " file(s) have been splitted into " + str(i) + "files")

    def get_filename(self, data_idx):
        return os.path.join(self.source_dir, self.name+"_splitted", self.name+"_splitted_"+str(data_idx)+ ".tfrecords")


def extract_vectorized(map_features: List[waymo_map_pb2.MapFeature], map_name) -> vectorized_map_pb2.VectorizedMap:
    vec_map = vectorized_map_pb2.VectorizedMap()
    vec_map.name = map_name
    vec_map.shifted_origin.x = 0.0
    vec_map.shifted_origin.y = 0.0
    vec_map.shifted_origin.z = 0.0
    max_pt = vectorized_map_pb2.Point()
    max_pt.x = 0.0
    max_pt.y = 0.0
    max_pt.z = 0.0
    min_pt = vectorized_map_pb2.Point()
    min_pt.x = 0.0
    min_pt.y = 0.0
    min_pt.z = 0.0

    boundaries = {}
    for map_feature in tqdm(map_features, desc="Extracting road boundaries"):
        if map_feature.WhichOneof("feature_data") == "road_line":
            boundaries[map_feature.id] = map_feature.road_line.polyline
        elif map_feature.WhichOneof("feature_data") == "road_edge":
            boundaries[map_feature.id] = map_feature.road_edge.polyline

    for map_feature in tqdm(map_features, desc="Converting the waymo map features into vector map"):
        new_element: vectorized_map_pb2.MapElement = vec_map.elements.add()
        new_element.id = bytes(map_feature.id)
        if map_feature.WhichOneof("feature_data") == "lane":
            temp_max_pt, temp_min_pt = translate_lane(new_element.road_lane, map_feature.lane, boundaries)
        elif map_feature.WhichOneof("feature_data") == "crosswalk":
            temp_max_pt, temp_min_pt = translate_crosswalk(new_element.ped_crosswalk, map_feature.crosswalk)
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


def translate_poly_line(ret: vectorized_map_pb2.Polyline, polyline: List[waymo_map_pb2.MapPoint]) -> (List[float], List[float]):
    points = [[point.x, point.y, point.z] for point in polyline]
    shifted_points = [[0.0, 0.0, 0.0]] + points
    shifted_points.pop(-1)

    points = np.array(points)
    shifted_points = np.array(shifted_points)
    max_pt = np.max(points, axis=0)
    min_pt = np.min(points, axis=0)
    ret_polyline = (np.array(points) - np.array(shifted_points)) * 1000
    ret_dx = (ret_polyline[:, 0])
    ret_dy = (ret_polyline[:, 1])
    ret_dz = (ret_polyline[:, 2])
    ret_h = np.arctan(ret_dx/ret_dy)
    ret.dx_mm.extend(ret_dx.astype(int).tolist())
    ret.dy_mm.extend(ret_dy.astype(int).tolist())
    ret.dz_mm.extend(ret_dz.astype(int).tolist())
    ret.h_rad.extend(ret_h)

    return max_pt, min_pt


def translate_lane(road_lane: vectorized_map_pb2.RoadLane, lane: waymo_map_pb2.LaneCenter, boundaries: Dict) -> (List[float], List[float]):
    center_max, center_min = translate_poly_line(road_lane.center, lane.polyline)
    left_max = center_max
    right_max = center_max
    left_min = center_min
    right_min = center_min
    if lane.left_boundaries:
        left_boundary_map_pts: List[waymo_map_pb2.MapPoint] = []

        for left_boundary in lane.left_boundaries:
            left_boundary_map_pts.extend(boundaries[left_boundary.boundary_feature_id])
        left_max, left_min = translate_poly_line(road_lane.left_boundary, left_boundary_map_pts)
    if lane.right_boundaries:
        right_boundary_map_pts: List[waymo_map_pb2.MapPoint] = []
        for right_boundary in lane.right_boundaries:
            right_boundary_map_pts.extend(boundaries[right_boundary.boundary_feature_id])
        right_max, right_min = translate_poly_line(road_lane.right_boundary, right_boundary_map_pts)

    road_lane.entry_lanes.extend(np.array(lane.entry_lanes).astype(bytes).tolist())
    road_lane.exit_lanes.extend(np.array(lane.exit_lanes).astype(bytes).tolist())
    for neighbor in lane.left_neighbors:
        road_lane.adjacent_lanes_left.append(bytes(neighbor.feature_id))
    for neighbor in lane.right_neighbors:
        road_lane.adjacent_lanes_right.append(bytes(neighbor.feature_id))
    max_point = np.max([center_max, left_max, right_max], axis=0)
    min_point = np.min([center_min, left_min, right_min], axis=0)
    return max_point, min_point


def translate_crosswalk(ret: vectorized_map_pb2.PedCrosswalk, lane: waymo_map_pb2.Crosswalk) -> (List[int], List[int]):
    max_pt, min_pt = translate_poly_line(ret.polygon, lane.polygon)
    return max_pt, min_pt


def extract_traffic_lights(dynamic_states: List[scenario_pb2.DynamicMapState]) -> Dict[Tuple[int, int], TrafficLightStatus]:
    ret: Dict[Tuple[int, int], TrafficLightStatus] = {}
    for i, dynamic_state in enumerate(dynamic_states):
        for lane_state in dynamic_state.lane_states:
            ret[(lane_state.lane, i)] = translate_traffic_state(lane_state.state)
    return ret


def translate_traffic_state(state: waymo_map_pb2.TrafficSignalLaneState.State) -> TrafficLightStatus:
    # The traffic light type doesn't align between waymo and trajdata,
    # but I think trajdata TrafficLightStatus should include yellow light
    # for now I let caution = red
    if state in GREEN:
        return TrafficLightStatus.GREEN
    if state in RED:
        return TrafficLightStatus.RED
    return TrafficLightStatus.UNKNOWN


def interpolate_array(data: np.array) -> np.array:
    interpolated_series = pd.Series(data)
    first_non_zero = interpolated_series.ne(0).idxmax()
    last_non_zero = interpolated_series.ne(0)[::-1].idxmax()
    # Apply linear interpolation to the internal zeros
    interpolated_series[first_non_zero:last_non_zero] = \
        interpolated_series[first_non_zero:last_non_zero].replace(0, np.nan).interpolate()
    return interpolated_series.values
