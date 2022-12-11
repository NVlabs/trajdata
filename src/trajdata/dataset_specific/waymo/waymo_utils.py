import pathlib
from typing import List, Final, Tuple, Dict
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import copy
from multiprocessing import Pool
from waymo_open_dataset.protos import scenario_pb2, map_pb2 as waymo_map_pb2

from trajdata.maps import TrafficLightStatus
from trajdata.proto import vectorized_map_pb2


WAYMO_DT: Final[float] = 0.1
SOURCE_DIR = "../../../../../scenarios"
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
    FixedExtent, VariableExtent
)

def parse_data(data):
    scenario = scenario_pb2.Scenario()
    scenario.ParseFromString(data)
    return scenario
class WaymoScenarios:
    def __init__(self, dataset_name, source_dir=SOURCE_DIR, load=True, num_parallel_reads=None):
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


# way = WaymoScenarios(dataset_name='haha')

def extract_vectorized(map_features: List[waymo_map_pb2.MapFeature], map_name) -> vectorized_map_pb2.VectorizedMap:
    vec_map = vectorized_map_pb2.VectorizedMap()
    vec_map.name = map_name
    max_pt: vectorized_map_pb2.Point
    max_x = np.max()
    min_pt: vectorized_map_pb2.Point
    for map_feature in map_features:
        new_element: vectorized_map_pb2.MapElement = vec_map.elements.add()
        new_element.id = map_feature.id
        if map_feature.HasField("lane"):
            new_element.road_lane = translate_lane(map_feature.lane)
        if map_feature.HasField("crosswalk"):
            new_element.ped_crosswalk = translate_crosswalk(map_feature.crosswalk)
    return vec_map

def translate_agent_type(agent_type):
    if agent_type == scenario_pb2.Track.ObjectType.TYPE_VEHICLE:
        return AgentType.VEHICLE
    if agent_type == scenario_pb2.Track.ObjectType.TYPE_PEDESTRIAN:
        return AgentType.PEDESTRIAN
    if agent_type == scenario_pb2.Track.ObjectType.TYPE_CYCLIST:
        return AgentType.BICYCLE
    if agent_type == scenario_pb2.Track.ObjectType.OTHER:
        return AgentType.UNKNOWN
    return -1


def translate_poly_line(polyline: List[waymo_map_pb2.MapPoint]) -> vectorized_map_pb2.Polyline:
    ret = vectorized_map_pb2.Polyline()
    for point in polyline:
        ret.dx_mm.add(round(point.x * 100))
        ret.dy_mm.add(round(point.y * 100))
        ret.dz_mm.add(round(point.z * 100))
    return ret


def translate_lane(lane: waymo_map_pb2.LaneCenter) -> vectorized_map_pb2.RoadLane:
    ret = vectorized_map_pb2.RoadLane
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
    ret.center = translate_poly_line(center)
    ret.left_boundary = translate_poly_line(left_boundary)
    ret.right_boundary = translate_poly_line(right_boundary)

    ret.entry_lanes[:] = lane.entry_lanes.copy()
    ret.exit_lanes[:] = lane.exit_lanes.copy()
    ret.adjacent_lanes_left[:] = [neighbor.feature_id for neighbor in lane.left_neighbors]
    ret.adjacent_lanes_right[:] = [neighbor.feature_id for neighbor in lane.right_neighbors]
    return ret


def translate_crosswalk(lane: waymo_map_pb2.Crosswalk) -> vectorized_map_pb2.PedCrosswalk:
    ret = vectorized_map_pb2.PedCrosswalk()
    ret.polygon = translate_poly_line(lane.polygon)
    return ret

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

# agent_list: List[AgentMetadata] = []
# agent_presence: List[List[AgentMetadata]] = [
#     [] for _ in range(91)
# ]
# scenario = load_tfrecords(data_dir + '/training', False)[0]
# agent_ids = []
# agent_translations = []
# agent_velocities = []
# agent_yaws = []
# agent_ml_class = []
# agent_sizes = []
#
# for index, track in enumerate(scenario.tracks):
#     agent_name = track.id
#     if index == scenario.sdc_track_index:
#         agent_name = "ego"
#
#     agent_ids.append(agent_name)
#
#     agent_type: AgentType = translate_agent_type(track.object_type)
#     agent_ml_class.append(agent_type)
#     states = track.states
#     translations = [[state.center_x, state.center_y, state.center_z] for state in states]
#     agent_translations.extend(translations)
#     velocities = [[state.velocity_x, state.velocity_y] for state in states]
#     agent_velocities.extend(velocities)
#     sizes = [[state.length, state.width, state.height] for state in states]
#     agent_sizes.extend(sizes)
#     yaws = [state.heading for state in states]
#     agent_yaws.extend(yaws)
#
#     first_timestep = 0
#     states = track.states
#     for timestep in range(91):
#         if states[timestep].valid:
#             first_timestep = timestep
#             break
#     last_timestep = 90
#     for timestep in range(91):
#         if states[90 - timestep].valid:
#             last_timestep = timestep
#             break
#
#     agent_info = AgentMetadata(
#         name=agent_name,
#         agent_type=agent_type,
#         first_timestep=first_timestep,
#         last_timestep=last_timestep,
#         extent=VariableExtent(),
#     )
#     if last_timestep - first_timestep != 0:
#         agent_list.append(agent_info)
#
#     for timestep in range(first_timestep, last_timestep + 1):
#         agent_presence[timestep].append(agent_info)
#
# agent_ids = np.repeat(agent_ids, 91)
#
# agent_translations = np.array(agent_translations)
# agent_velocities = np.array(agent_velocities)
# agent_sizes = np.array(agent_sizes)
#
# agent_ml_class = np.repeat(agent_ml_class, 91)
# agent_yaws = np.array(agent_yaws)
#
# print(agent_ids.shape)
# print(agent_translations.shape)
# print(agent_velocities.shape)
# print(agent_sizes.shape)
# print(agent_ml_class.shape)
# print(agent_yaws.shape)
#
# all_agent_data = np.concatenate(
#     [
#         agent_translations,
#         agent_velocities,
#         np.expand_dims(agent_yaws, axis=1),
#         np.expand_dims(agent_ml_class, axis=1),
#         agent_sizes,
#     ],
#     axis=1,
# )
#
# traj_cols = ["x", "y", "z", "vx", "vy", "heading"]
# class_cols = ["class_id"]
# extent_cols = ["length", "width", "height"]
# agent_frame_ids = np.resize(
#     np.arange(91), 63*91
# )
#
# all_agent_data_df = pd.DataFrame(
#     all_agent_data,
#     columns=traj_cols + class_cols + extent_cols,
#     index=[agent_ids, agent_frame_ids],
# )
#
# all_agent_data_df.index.names = ["agent_id", "scene_ts"]
# all_agent_data_df.sort_index(inplace=True)
# all_agent_data_df.reset_index(level=1, inplace=True)
#
# all_agent_data_df[["ax", "ay"]] = (
#         arr_utils.agent_aware_diff(
#             all_agent_data_df[["vx", "vy"]].to_numpy(), agent_ids
#         )
#         / WAYMO_DT
# )
# final_cols = [
#                  "x",
#                  "y",
#                  "vx",
#                  "vy",
#                  "ax",
#                  "ay",
#                  "heading",
#              ] + extent_cols
# all_agent_data_df.reset_index(inplace=True)
# all_agent_data_df["agent_id"] = all_agent_data_df["agent_id"].astype(str)
# all_agent_data_df.set_index(["agent_id", "scene_ts"], inplace=True)
#
# print(all_agent_data_df)
# print(all_agent_data_df.columns)
# print(all_agent_data_df.loc[:, final_cols])
# print(pd.concat([all_agent_data_df.loc[:, final_cols]]))
# print(scenario.tracks[0].id)
# print(scenario.tracks[0].states[1].height)

# for track in scenario.tracks:
#
#     print(all_agent_data_df['height'][str(track.id)][0])
#     break
