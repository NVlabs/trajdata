import os.path
import pathlib
from typing import Dict, Final, List, Tuple
from subprocess import check_call, check_output
import numpy as np
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

class WaymoScenarios:
    def __init__(self, dataset_name, source_dir, download=True, split=True):
        if dataset_name not in WAYMO_DATASET_NAMES:
            raise RuntimeError('Wrong dataset name. Please choose name from '+str(WAYMO_DATASET_NAMES))
        self.name = dataset_name
        self.source_dir = source_dir
        self.split = split
        if dataset_name in ["training", "validation", "validation_interactive"]:
            self.scene_length = 9
        elif dataset_name in ["testing", "testing_interactive"]:
            self.scene_length = 1
        elif dataset_name in ["training_20s"]:
            self.scene_length = 20
        self.num_scenarios = 44920
        if download:
            self.download_dataset()
        if split:
            self.split_scenarios()

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
            print(str(i) + " scenarios from " + str(len(file_names)) + " file(s) have been splitted into "+ str(i) + "files")

    def get_filename(self, data_idx):
        return os.path.join(self.source_dir, self.name+"_splitted", self.name+"_splitted_"+str(data_idx)+ ".tfrecords")


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
            road_lane, temp_max_pt, temp_min_pt = translate_lane(map_feature.lane, boundaries)
            new_element.road_lane.CopyFrom(road_lane)
        elif map_feature.WhichOneof("feature_data") == "crosswalk":
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


def translate_poly_line(polyline: List[waymo_map_pb2.MapPoint]) -> (vectorized_map_pb2.Polyline, List[int], List[int]):
    ret = vectorized_map_pb2.Polyline()
    max_pt = [0.0, 0.0, 0.0]
    min_pt = [0.0, 0.0, 0.0]
    for idx, point in enumerate(polyline):
        if idx == 0:
            ret.dx_mm.append(round(point.x * 100))
            ret.dy_mm.append(round(point.y * 100))
            ret.dz_mm.append(round(point.z * 100))
        else:
            ret.dx_mm.append(round(point.x * 100) - ret.dx_mm[idx - 1])
            ret.dy_mm.append(round(point.y * 100) - ret.dy_mm[idx - 1])
            ret.dz_mm.append(round(point.z * 100) - ret.dz_mm[idx - 1])
        ret.h_rad.append(np.arctan(point.y / point.x))
        max_pt = get_larger_elems(max_pt, [point.x, point.y, point.z])
        min_pt = get_smaller_elems(min_pt, [point.x, point.y, point.z])
    return ret, max_pt, min_pt


def translate_lane(lane: waymo_map_pb2.LaneCenter, boundaries: Dict) -> (vectorized_map_pb2.RoadLane, List[int], List[int]):
    road_lane = vectorized_map_pb2.RoadLane()
    center, center_max, center_min = translate_poly_line(lane.polyline)
    left_max = center_max
    right_max = center_max
    left_min = center_min
    right_min = center_min
    road_lane.center.CopyFrom(center)
    if lane.left_boundaries:
        left_boundary_map_pts: List[waymo_map_pb2.MapPoint] = []

        for left_boundary in lane.left_boundaries:
            left_boundary_map_pts.extend(boundaries[left_boundary.boundary_feature_id])
        left_boundary, left_max, left_min = translate_poly_line(left_boundary_map_pts)
        road_lane.left_boundary.CopyFrom(left_boundary)
    if lane.right_boundaries:
        right_boundary_map_pts: List[waymo_map_pb2.MapPoint] = []
        for right_boundary in lane.right_boundaries:
            right_boundary_map_pts.extend(boundaries[right_boundary.boundary_feature_id])
        right_boundary, right_max, right_min = translate_poly_line(right_boundary_map_pts)
        road_lane.right_boundary.CopyFrom(right_boundary)

    for entry_lane in lane.entry_lanes:
        road_lane.entry_lanes.append(bytes(entry_lane))
    for exit_lane in lane.exit_lanes:
        road_lane.exit_lanes.append(bytes(exit_lane))
    for neighbor in lane.left_neighbors:
        road_lane.adjacent_lanes_left.append(bytes(neighbor.feature_id))
    for neighbor in lane.right_neighbors:
        road_lane.adjacent_lanes_right.append(bytes(neighbor.feature_id))
    max_point = [np.max([center_max[i], left_max[i], right_max[i]]) for i in range(3)]
    min_point = [np.min([center_min[i], left_min[i], right_min[i]]) for i in range(3)]
    return road_lane, max_point, min_point


def translate_crosswalk(lane: waymo_map_pb2.Crosswalk) -> (vectorized_map_pb2.PedCrosswalk, List[int], List[int]):
    ret = vectorized_map_pb2.PedCrosswalk()
    polygon, max_pt, min_pt = translate_poly_line(lane.polygon)
    ret.polygon.CopyFrom(polygon)
    return ret, max_pt, min_pt


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
