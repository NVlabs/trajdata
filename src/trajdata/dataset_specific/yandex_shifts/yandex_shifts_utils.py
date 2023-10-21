import os
from collections import defaultdict
from typing import Dict, List, Tuple, Union, Any
import numpy as np
import pandas as pd
from ysdc_dataset_api.proto import Scene as YSDCScene
from ysdc_dataset_api.proto.map_pb2 import PathGraph as YSDCPathGraph
from ysdc_dataset_api.proto.dataset_pb2 import VehicleTrack as YSDCVehicleTrack
from trajdata.maps import TrafficLightStatus, VectorMap
from trajdata.maps.vec_map_elements import PedCrosswalk, Polyline, RoadLane, RoadArea
from trajdata.data_structures.batch_element import AgentBatchElement, SceneBatchElement
from trajdata.data_structures.agent import AgentMetadata, AgentType, VariableExtent
from trajdata.data_structures import Scene as TRAJScene


YSDC_DT = 0.2
YSDC_LENGTH = 50


def fetch_season_info(
    batch_element: Union[AgentBatchElement, SceneBatchElement]
) -> int:
    return batch_element.track_info["season"]


def fetch_day_time_info(
    batch_element: Union[AgentBatchElement, SceneBatchElement]
) -> int:
    return batch_element.track_info["day_time"]


def fetch_track_location_info(
    batch_element: Union[AgentBatchElement, SceneBatchElement]
) -> int:
    return batch_element.track_info["track_location"]


def fetch_sun_phase_info(
    batch_element: Union[AgentBatchElement, SceneBatchElement]
) -> int:
    return batch_element.track_info["sun_phase"]


def fetch_precipitation_info(
    batch_element: Union[AgentBatchElement, SceneBatchElement]
) -> int:
    return batch_element.track_info["precipitation"]


def read_scene_from_original_proto(path: str) -> YSDCScene:
    with open(path, "rb") as f:
        scene = YSDCScene()
        scene.ParseFromString(f.read())
    return scene


def get_scene_path(data_dir: str, scene_idx: int) -> str:
    return (
        os.path.join(data_dir, str(scene_idx // 1000).zfill(3), str(scene_idx).zfill(6))
        + ".pb"
    )


def fix_headings(agents_data_df: pd.DataFrame) -> pd.DataFrame:
    headings = agents_data_df["heading"].values
    previous_headings = np.roll(headings, 1)
    previous_headings[0] = headings[0]
    normalized_angle_diff = np.abs(headings - previous_headings) % (2 * np.pi)
    fixed_headings = np.where(
        np.minimum(normalized_angle_diff, 2 * np.pi - normalized_angle_diff)
        > np.pi / 2,
        headings + np.pi,
        headings,
    )
    agents_data_df["heading"] = fixed_headings
    return agents_data_df


def fill_missing_timestamps(
    agents_data_df: pd.DataFrame, agent_id_to_time_range: dict
) -> pd.DataFrame:
    filled_agents_data_df = []
    for agent_id, agent_df in agents_data_df.groupby("agent_id"):
        state_idx = 0
        for ts in range(
            agent_id_to_time_range[agent_id][0], agent_id_to_time_range[agent_id][1] + 1
        ):
            if (
                state_idx < agent_df.shape[0]
                and agent_df.iloc[state_idx]["scene_ts"] == ts
            ):
                d = agent_df.iloc[state_idx].to_dict()
                d["agent_id"] = agent_id
                filled_agents_data_df.append(d)
                state_idx += 1
            else:
                filled_agents_data_df.append(
                    {
                        "agent_id": agent_id,
                        "scene_ts": ts,
                        "x": None,
                        "y": None,
                        "z": None,
                        "vx": None,
                        "vy": None,
                        "ax": None,
                        "ay": None,
                        "heading": None,
                        "length": None,
                        "width": None,
                        "height": None,
                    }
                )
    return pd.DataFrame(filled_agents_data_df).sort_values(by=["agent_id", "scene_ts"])


def map_ysdc_to_trajdata_traffic_light_status(
    ysdc_tl_status: int,
) -> TrafficLightStatus:
    mapping = {
        -1: TrafficLightStatus.NO_DATA,
        0: TrafficLightStatus.UNKNOWN,
        1: TrafficLightStatus.GREEN,
        2: TrafficLightStatus.GREEN,
        3: TrafficLightStatus.RED,
        4: TrafficLightStatus.RED,
        5: TrafficLightStatus.RED,
        6: TrafficLightStatus.UNKNOWN,
        7: TrafficLightStatus.UNKNOWN,
        8: TrafficLightStatus.UNKNOWN,
        9: TrafficLightStatus.UNKNOWN,
        10: TrafficLightStatus.UNKNOWN,
        11: TrafficLightStatus.RED,
    }
    return mapping[ysdc_tl_status]


def extract_traffic_light_status(
    ysdc_scene: YSDCScene,
) -> Dict[Tuple[str, int], TrafficLightStatus]:
    traffic_light_data = {}
    n_states = len(ysdc_scene.past_vehicle_tracks) + len(
        ysdc_scene.future_vehicle_tracks
    )
    traffic_light_section_id_to_state = {}
    for traffic_light in ysdc_scene.traffic_lights:
        for traffic_light_section in traffic_light.sections:
            traffic_light_section_id_to_state[
                traffic_light_section.id
            ] = traffic_light_section.state
    for lane_idx, lane in enumerate(ysdc_scene.path_graph.lanes):
        # YSDC dataset supports also left_section_id and right_section_id
        conventional_lane_id = f"lane_{lane_idx}"
        lane_main_section_id = lane.traffic_light_section_ids.main_section_id
        if lane_main_section_id not in traffic_light_section_id_to_state:
            traffic_light_section_id_to_state[lane_main_section_id] = -1
        ysdc_traffic_light_state = traffic_light_section_id_to_state[
            lane_main_section_id
        ]
        conventional_traffic_light_state = map_ysdc_to_trajdata_traffic_light_status(
            ysdc_traffic_light_state
        )
        for ts in range(n_states):
            traffic_light_data[
                (conventional_lane_id, ts)
            ] = conventional_traffic_light_state
    return traffic_light_data


def extract_vectorized(map_features: YSDCPathGraph, map_name: str) -> VectorMap:
    vec_map = VectorMap(map_id=map_name)
    max_pt = np.array([np.nan, np.nan])
    min_pt = np.array([np.nan, np.nan])

    for lane_idx, lane in enumerate(map_features.lanes):
        lane_centers = np.array([(node.x, node.y) for node in lane.centers])
        max_pt = np.nanmax(np.vstack([max_pt, lane_centers]), axis=0)
        min_pt = np.nanmin(np.vstack([min_pt, lane_centers]), axis=0)
        vec_map.add_map_element(
            RoadLane(
                # YSDC only has center lane
                id=f"lane_{lane_idx}",
                center=Polyline(lane_centers),
            )
        )

    for crosswalk_idx, crosswalk in enumerate(map_features.crosswalks):
        crosswalk_points = np.array(
            [(node.x, node.y) for node in crosswalk.geometry.points]
        )
        max_pt = np.nanmax(np.vstack([max_pt, crosswalk_points]), axis=0)
        min_pt = np.nanmin(np.vstack([min_pt, crosswalk_points]), axis=0)
        vec_map.add_map_element(
            PedCrosswalk(
                id=f"crosswalk_{crosswalk_idx}", polygon=Polyline(crosswalk_points)
            )
        )

    for road_polygon_idx, road_polygon in enumerate(map_features.road_polygons):
        road_polygon_points = np.array(
            [(node.x, node.y) for node in road_polygon.geometry.points]
        )
        max_pt = np.nanmax(np.vstack([max_pt, road_polygon_points]), axis=0)
        min_pt = np.nanmin(np.vstack([min_pt, road_polygon_points]), axis=0)
        vec_map.add_map_element(
            RoadArea(
                id=f"road_polygon_{road_polygon_idx}",
                exterior_polygon=Polyline(road_polygon_points),
            )
        )

    vec_map.extent = np.array([*min_pt, 0, *max_pt, 0])
    return vec_map


def prepare_agent_info_dict_from_track(
    track: YSDCVehicleTrack, scene_ts: int, entity: AgentType, is_ego: bool = False
) -> Dict[str, Any]:
    assert entity in [AgentType.VEHICLE, AgentType.PEDESTRIAN]
    return {
        "agent_id": "ego" if is_ego else str(track.track_id),
        "scene_ts": scene_ts,
        "x": track.position.x,
        "y": track.position.y,
        "z": track.position.z,
        "vx": track.linear_velocity.x,
        "vy": track.linear_velocity.y,
        "ax": 0 if entity == AgentType.PEDESTRIAN else track.linear_acceleration.x,
        "ay": 0 if entity == AgentType.PEDESTRIAN else track.linear_acceleration.y,
        "heading": np.arctan2(track.linear_velocity.y, track.linear_velocity.x)
        if entity == AgentType.PEDESTRIAN
        else track.yaw,
        "length": track.dimensions.x,
        "width": track.dimensions.y,
        "height": track.dimensions.z,
    }


def update_time_range(
    agent_id: str,
    timestamp: int,
    agent_id_to_time_range: Dict[str, Tuple[float, float]],
) -> None:
    agent_id_to_time_range[agent_id] = (
        min(agent_id_to_time_range[agent_id][0], timestamp),
        max(agent_id_to_time_range[agent_id][1], timestamp),
    )


def extract_agent_data_from_ysdc_scene(
    ysdc_scene: YSDCScene, trajdata_scene: TRAJScene
) -> Tuple[pd.DataFrame, List[AgentMetadata], List[List[AgentMetadata]]]:
    agent_list: List[AgentMetadata] = []
    agent_presence: List[List[AgentMetadata]] = [
        [] for _ in range(trajdata_scene.length_timesteps)
    ]
    scene_agents_data = defaultdict(list)
    agent_id_to_time_range = defaultdict(lambda: (np.inf, -np.inf))
    agent_id_to_type = {"ego": AgentType.VEHICLE}
    agents_types_data = [
        (
            AgentType.VEHICLE,
            list(ysdc_scene.past_vehicle_tracks)
            + list(ysdc_scene.future_vehicle_tracks),
        ),
        (
            AgentType.PEDESTRIAN,
            list(ysdc_scene.past_pedestrian_tracks)
            + list(ysdc_scene.future_pedestrian_tracks),
        ),
    ]
    ego_agent_data = list(ysdc_scene.past_ego_track) + list(ysdc_scene.future_ego_track)
    for agent_type, scene_moment_states in agents_types_data:
        for timestamp, scene_moment_state in enumerate(scene_moment_states):
            for agent_moment_state in scene_moment_state.tracks:
                agent_info_dict = prepare_agent_info_dict_from_track(
                    agent_moment_state, timestamp, agent_type
                )
                scene_agents_data[agent_info_dict["agent_id"]].append(agent_info_dict)
                update_time_range(
                    agent_info_dict["agent_id"], timestamp, agent_id_to_time_range
                )
                agent_id_to_type[str(agent_moment_state.track_id)] = agent_type
    for timestamp, ego_agent_moment_state in enumerate(ego_agent_data):
        agent_info_dict = prepare_agent_info_dict_from_track(
            ego_agent_moment_state, timestamp, AgentType.VEHICLE, True
        )
        scene_agents_data[agent_info_dict["agent_id"]].append(agent_info_dict)
        update_time_range(
            agent_info_dict["agent_id"], timestamp, agent_id_to_time_range
        )
    scene_agents_data_df = pd.DataFrame(
        [item for sublist in scene_agents_data.values() for item in sublist]
    ).sort_values(by=["agent_id", "scene_ts"])
    scene_agents_data_df = fix_headings(scene_agents_data_df)
    scene_agents_data_df = fill_missing_timestamps(
        scene_agents_data_df, agent_id_to_time_range
    )
    scene_agents_data_df = (
        scene_agents_data_df.groupby("agent_id", group_keys=True)
        .apply(lambda group: group.interpolate(limit_area="inside"))
        .reset_index(drop=True)
        .set_index(["agent_id", "scene_ts"])
    )
    for agent_id in agent_id_to_type.keys():
        agent_list.append(
            AgentMetadata(
                name=agent_id,
                agent_type=agent_id_to_type[agent_id],
                first_timestep=agent_id_to_time_range[agent_id][0],
                last_timestep=agent_id_to_time_range[agent_id][1],
                extent=VariableExtent(),
            )
        )
        for ts in range(
            agent_id_to_time_range[agent_id][0], agent_id_to_time_range[agent_id][1] + 1
        ):
            agent_presence[ts].append(
                AgentMetadata(
                    name=agent_id,
                    agent_type=agent_id_to_type[agent_id],
                    first_timestep=agent_id_to_time_range[agent_id][0],
                    last_timestep=agent_id_to_time_range[agent_id][1],
                    extent=VariableExtent(),
                )
            )
    return scene_agents_data_df, agent_list, agent_presence
