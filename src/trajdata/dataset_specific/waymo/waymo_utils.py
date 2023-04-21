import os
from pathlib import Path
from subprocess import check_call, check_output
from typing import Dict, Final, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from intervaltree import Interval, IntervalTree
from tqdm import tqdm
from waymo_open_dataset.protos import map_pb2 as waymo_map_pb2
from waymo_open_dataset.protos import scenario_pb2

from trajdata.maps import TrafficLightStatus
from trajdata.proto import vectorized_map_pb2

WAYMO_DT: Final[float] = 0.1
WAYMO_DATASET_NAMES = [
    "testing",
    "testing_interactive",
    "training",
    "training_20s",
    "validation",
    "validation_interactive",
]

TRAIN_SCENE_LENGTH = 91
VAL_SCENE_LENGTH = 91
TEST_SCENE_LENGTH = 11
TRAIN_20S_SCENE_LENGTH = 201

GREEN = [
    waymo_map_pb2.TrafficSignalLaneState.State.LANE_STATE_ARROW_GO,
    waymo_map_pb2.TrafficSignalLaneState.State.LANE_STATE_GO,
]
RED = [
    waymo_map_pb2.TrafficSignalLaneState.State.LANE_STATE_ARROW_CAUTION,
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
    def __init__(
        self,
        dataset_name: str,
        source_dir: Path,
        download: bool = False,
        split: bool = False,
    ):
        if dataset_name not in WAYMO_DATASET_NAMES:
            raise RuntimeError(
                "Wrong dataset name. Please choose name from "
                + str(WAYMO_DATASET_NAMES)
            )

        self.name = dataset_name
        self.source_dir = source_dir
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

        split_path = self.source_dir / (self.name + "_splitted")
        if split or not split_path.is_dir():
            self.split_scenarios()
        else:
            self.num_scenarios = len(os.listdir(split_path))

    def download_dataset(self) -> None:
        # check_call("snap install google-cloud-sdk --classic".split())
        gsutil = check_output(["which", "gsutil"])
        download_cmd = (
            str(gsutil.decode("utf-8"))
            + "-m cp -r gs://waymo_open_dataset_motion_v_1_1_0/uncompressed/scenario/"
            + str(self.name)
            + " "
            + str(self.source_dir)
        ).split()
        check_call(download_cmd)

    def split_scenarios(
        self, num_parallel_reads: int = 20, verbose: bool = True
    ) -> None:
        source_it: Path = (self.source_dir / self.name).glob("*")
        file_names: List[str] = [str(file_name) for file_name in source_it]
        if verbose:
            print("Loading tfrecord files...")
        dataset = tf.data.TFRecordDataset(
            file_names, compression_type="", num_parallel_reads=num_parallel_reads
        )

        if verbose:
            print("Splitting tfrecords...")

        splitted_dir: Path = self.source_dir / f"{self.name}_splitted"
        if not splitted_dir.exists():
            splitted_dir.mkdir(parents=True)

        scenario_num: int = 0
        for data in tqdm(dataset):
            file_name: Path = (
                splitted_dir / f"{self.name}_splitted_{scenario_num}.tfrecords"
            )
            with tf.io.TFRecordWriter(str(file_name)) as file_writer:
                file_writer.write(data.numpy())

            scenario_num += 1

        self.num_scenarios = scenario_num
        if verbose:
            print(
                str(self.num_scenarios)
                + " scenarios from "
                + str(len(file_names))
                + " file(s) have been split into "
                + str(self.num_scenarios)
                + " files."
            )

    def get_filename(self, data_idx):
        return (
            self.source_dir
            / f"{self.name}_splitted"
            / f"{self.name}_splitted_{data_idx}.tfrecords"
        )


def extract_vectorized(
    map_features: List[waymo_map_pb2.MapFeature], map_name: str, verbose: bool = False
) -> vectorized_map_pb2.VectorizedMap:
    vec_map = vectorized_map_pb2.VectorizedMap()
    vec_map.name = map_name
    vec_map.shifted_origin.x = 0.0
    vec_map.shifted_origin.y = 0.0
    vec_map.shifted_origin.z = 0.0

    max_pt = np.array([np.nan, np.nan, np.nan])
    min_pt = np.array([np.nan, np.nan, np.nan])

    boundaries = {}
    for map_feature in tqdm(
        map_features, desc="Extracting road boundaries", disable=not verbose
    ):
        if map_feature.WhichOneof("feature_data") == "road_line":
            boundaries[map_feature.id] = map_feature.road_line.polyline
        elif map_feature.WhichOneof("feature_data") == "road_edge":
            boundaries[map_feature.id] = map_feature.road_edge.polyline

    for map_feature in tqdm(
        map_features, desc="Extracting map elements", disable=not verbose
    ):
        if map_feature.WhichOneof("feature_data") == "lane":
            temp_max_pt, temp_min_pt, modified_lane_ids = translate_lane(
                vec_map, map_feature, boundaries
            )
        elif map_feature.WhichOneof("feature_data") == "crosswalk":
            new_element: vectorized_map_pb2.MapElement = vec_map.elements.add()
            new_element.id = bytes(map_feature.id)
            temp_max_pt, temp_min_pt = translate_crosswalk(
                new_element.ped_crosswalk, map_feature.crosswalk
            )
        else:
            continue

        max_pt = np.fmax(max_pt, temp_max_pt)
        min_pt = np.fmin(min_pt, temp_min_pt)

    (vec_map.max_pt.x, vec_map.max_pt.y, vec_map.max_pt.z) = max_pt

    (vec_map.min_pt.x, vec_map.min_pt.y, vec_map.min_pt.z) = min_pt

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


def translate_polyline(
    ret: vectorized_map_pb2.Polyline, polyline: List[waymo_map_pb2.MapPoint]
) -> Tuple[List[float], List[float]]:
    points = [[point.x, point.y, point.z] for point in polyline]
    shifted_points = [[0.0, 0.0, 0.0]] + points
    shifted_points.pop(-1)

    points = np.array(points)
    shifted_points = np.array(shifted_points)
    max_pt = np.max(points, axis=0)
    min_pt = np.min(points, axis=0)
    ret_polyline = (np.array(points) - np.array(shifted_points)) * 1000
    ret_dx = ret_polyline[:, 0]
    ret_dy = ret_polyline[:, 1]
    ret_dz = ret_polyline[:, 2]
    ret_h = np.arctan2(ret_dy, ret_dx)
    ret.dx_mm.extend(ret_dx.astype(int).tolist())
    ret.dy_mm.extend(ret_dy.astype(int).tolist())
    ret.dz_mm.extend(ret_dz.astype(int).tolist())
    ret.h_rad.extend(ret_h)

    return max_pt, min_pt


def is_full_boundary(lane_boundaries, num_lane_indices: int) -> bool:
    """Returns True if a given boundary is connected (there are no gaps)
    and every lane center index has a corresponding boundary point.

    Returns:
        bool
    """
    covers_all: bool = lane_boundaries[0].lane_start_index == 0 and lane_boundaries[
        0
    ].lane_end_index == (num_lane_indices - 1)
    for idx in range(1, len(lane_boundaries)):
        if (
            lane_boundaries[idx].lane_start_index
            != lane_boundaries[idx - 1].lane_end_index + 1
        ):
            covers_all = False
            break

    return covers_all


def _merge_interval_data(
    data1: Union[Tuple[str, int], str], data2: Union[Tuple[str, int], str]
) -> Tuple[str, Set[int], Set[int]]:
    if data1[0] == data2[0] == "none":
        return ("none", set(), set())

    if data1[0] == "none" and data2[0] != "none":
        return data2

    if data1[0] != "none" and data2[0] == "none":
        return data1

    if data1[0] != "none" and data2[0] != "none":
        return ("both", data1[1] | data2[1], data1[2] | data2[2])


def split_lane_into_chunks(
    lane: waymo_map_pb2.LaneCenter, boundaries: Dict[int, List[waymo_map_pb2.MapPoint]]
) -> List[Interval]:
    # The data here is (lane_edges_available, left_boundary_ids, right_boundary_ids).
    left_boundaries = [
        (
            b.lane_start_index,
            b.lane_end_index + 1,
            ("left", set([b.boundary_feature_id]), set()),
        )
        for b in lane.left_boundaries
    ]
    right_boundaries = [
        (
            b.lane_start_index,
            b.lane_end_index + 1,
            ("right", set(), set([b.boundary_feature_id])),
        )
        for b in lane.right_boundaries
    ]

    boundary_intervals = IntervalTree.from_tuples(
        left_boundaries
        + right_boundaries
        + [(0, len(lane.polyline), ("none", set(), set()))]
    )

    boundary_intervals.split_overlaps()

    boundary_intervals.merge_equals(data_reducer=_merge_interval_data)
    intervals: List[Interval] = sorted(boundary_intervals)

    if len(intervals) > 1:
        merged_intervals: List[Interval] = [intervals.pop(0)]
        for _ in range(len(intervals)):
            if (
                merged_intervals[-1].data == intervals[0].data
                and merged_intervals[-1].end == intervals[0].begin
            ):
                combined_data = (
                    merged_intervals[-1].data[0],
                    merged_intervals[-1].data[1] | intervals[0].data[1],
                    merged_intervals[-1].data[2] | intervals[0].data[2],
                )
                merged_intervals[-1] = Interval(
                    merged_intervals[-1].begin,
                    intervals[0].end,
                    combined_data,
                )
                intervals.pop(0)
            else:
                merged_intervals.append(intervals.pop(0))
        intervals = merged_intervals

    left_boundary_tree = IntervalTree.from_tuples(left_boundaries)
    right_boundary_tree = IntervalTree.from_tuples(right_boundaries)
    lane_chunk_data: List[Tuple] = []
    for interval in intervals:
        center_chunk = lane.polyline[interval.begin : interval.end]
        if interval.data[0] == "none":
            lane_chunk_data.append((center_chunk, None, None))
        elif interval.data[0] == "left":
            left_chunk = subselect_boundary(
                boundaries, center_chunk, interval, left_boundary_tree
            )
            lane_chunk_data.append((center_chunk, left_chunk, None))
        elif interval.data[0] == "right":
            right_chunk = subselect_boundary(
                boundaries, center_chunk, interval, right_boundary_tree
            )
            lane_chunk_data.append((center_chunk, None, right_chunk))
        elif interval.data[0] == "both":
            left_chunk = subselect_boundary(
                boundaries, center_chunk, interval, left_boundary_tree
            )
            right_chunk = subselect_boundary(
                boundaries, center_chunk, interval, right_boundary_tree
            )
            lane_chunk_data.append((center_chunk, left_chunk, right_chunk))
        else:
            raise ValueError()

    return lane_chunk_data


def subselect_boundary(
    boundaries: Dict[int, List[waymo_map_pb2.MapPoint]],
    lane_center: List[waymo_map_pb2.MapPoint],
    interval: Interval,
    boundary_tree: IntervalTree,
) -> List[waymo_map_pb2.MapPoint]:
    print()
    relevant_indices = boundary_tree


def translate_lane(
    vec_map: vectorized_map_pb2.VectorizedMap,
    map_feature: waymo_map_pb2.MapFeature,
    boundaries: Dict[int, List[waymo_map_pb2.MapPoint]],
) -> Tuple[List[float], List[float], Optional[Dict[int, List[bytes]]]]:
    lane: waymo_map_pb2.LaneCenter = map_feature.lane

    modified_lane_ids = None
    if lane.left_boundaries or lane.right_boundaries:
        # Waymo lane boundaries are... complicated. See
        # https://github.com/waymo-research/waymo-open-dataset/issues/389
        # for more information. For now, we split lanes into chunks which
        # have consistent lane boundaries (either both left and right,
        # one of them, or none).
        intervals = split_lane_into_chunks(lane, boundaries)
        for idx, interval in enumerate(intervals):
            new_element: vectorized_map_pb2.MapElement = vec_map.elements.add()
            new_element.id = bytes(f"{map_feature.id}_{idx}")
            road_lane: vectorized_map_pb2.RoadLane = new_element.road_lane

            max_point, min_point = translate_polyline(road_lane.center, lane.polyline)

            if idx == 0:
                road_lane.entry_lanes.extend(
                    np.array(lane.entry_lanes).astype(bytes).tolist()
                )
            else:
                road_lane.entry_lanes.append(bytes(f"{map_feature.id}_{idx-1}"))

            if idx == len(intervals) - 1:
                road_lane.exit_lanes.extend(
                    np.array(lane.exit_lanes).astype(bytes).tolist()
                )
            else:
                road_lane.exit_lanes.append(bytes(f"{map_feature.id}_{idx+1}"))

            for neighbor in lane.left_neighbors:
                road_lane.adjacent_lanes_left.append(bytes(neighbor.feature_id))

            for neighbor in lane.right_neighbors:
                road_lane.adjacent_lanes_right.append(bytes(neighbor.feature_id))

            if interval.data == "both":
                pass

        modified_lane_ids = {map_feature.id: intervals}

    else:
        new_element: vectorized_map_pb2.MapElement = vec_map.elements.add()
        new_element.id = bytes(map_feature.id)
        road_lane: vectorized_map_pb2.RoadLane = new_element.road_lane

        max_point, min_point = translate_polyline(road_lane.center, lane.polyline)

        road_lane.entry_lanes.extend(np.array(lane.entry_lanes).astype(bytes).tolist())
        road_lane.exit_lanes.extend(np.array(lane.exit_lanes).astype(bytes).tolist())

        for neighbor in lane.left_neighbors:
            road_lane.adjacent_lanes_left.append(bytes(neighbor.feature_id))

        for neighbor in lane.right_neighbors:
            road_lane.adjacent_lanes_right.append(bytes(neighbor.feature_id))

    return max_point, min_point, modified_lane_ids


def translate_crosswalk(
    ret: vectorized_map_pb2.PedCrosswalk, lane: waymo_map_pb2.Crosswalk
) -> Tuple[List[int], List[int]]:
    max_pt, min_pt = translate_polyline(ret.polygon, lane.polygon)
    return max_pt, min_pt


def extract_traffic_lights(
    dynamic_states: List[scenario_pb2.DynamicMapState],
) -> Dict[Tuple[int, int], TrafficLightStatus]:
    ret: Dict[Tuple[int, int], TrafficLightStatus] = {}
    for i, dynamic_state in enumerate(dynamic_states):
        for lane_state in dynamic_state.lane_states:
            ret[(lane_state.lane, i)] = translate_traffic_state(lane_state.state)

    return ret


def translate_traffic_state(
    state: waymo_map_pb2.TrafficSignalLaneState.State,
) -> TrafficLightStatus:
    # TODO(bivanovic): The traffic light type doesn't align between waymo and trajdata,
    # but I think trajdata TrafficLightStatus should include yellow light
    # for now I let caution = red
    if state in GREEN:
        return TrafficLightStatus.GREEN
    if state in RED:
        return TrafficLightStatus.RED
    return TrafficLightStatus.UNKNOWN


def interpolate_array(data: List) -> np.array:
    return pd.DataFrame(data).interpolate(limit_area="inside").to_numpy()
