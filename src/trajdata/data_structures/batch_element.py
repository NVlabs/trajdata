from collections import defaultdict
from math import sqrt
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from trajdata.caching import SceneCache
from trajdata.data_structures.agent import AgentMetadata, AgentType
from trajdata.data_structures.scene import SceneTime, SceneTimeAgent
from trajdata.data_structures.state import StateArray
from trajdata.maps import MapAPI, RasterizedMapPatch, VectorMap
from trajdata.utils.state_utils import convert_to_frame_state, transform_from_frame


class AgentBatchElement:
    """A single element of an agent-centric batch."""

    # @profile
    def __init__(
        self,
        cache: SceneCache,
        data_index: int,
        scene_time_agent: SceneTimeAgent,
        history_sec: Tuple[Optional[float], Optional[float]],
        future_sec: Tuple[Optional[float], Optional[float]],
        agent_interaction_distances: Dict[
            Tuple[AgentType, AgentType], float
        ] = defaultdict(lambda: np.inf),
        incl_robot_future: bool = False,
        incl_raster_map: bool = False,
        raster_map_params: Optional[Dict[str, Any]] = None,
        map_api: Optional[MapAPI] = None,
        vector_map_params: Optional[Dict[str, Any]] = None,
        state_format: Optional[str] = None,
        standardize_data: bool = False,
        standardize_derivatives: bool = False,
        max_neighbor_num: Optional[int] = None,
    ) -> None:
        self.cache: SceneCache = cache
        self.data_index: int = data_index
        self.dt: float = scene_time_agent.scene.dt
        self.scene_ts: int = scene_time_agent.ts
        self.history_sec = history_sec
        self.future_sec = future_sec

        agent_info: AgentMetadata = scene_time_agent.agent
        self.agent_name: str = agent_info.name
        self.agent_type: AgentType = agent_info.type
        self.max_neighbor_num = max_neighbor_num

        raw_state: StateArray = cache.get_raw_state(agent_info.name, self.scene_ts)
        if state_format is not None:
            self.curr_agent_state_np = raw_state.as_format(state_format)
        else:
            self.curr_agent_state_np = raw_state

        self.standardize_data = standardize_data
        if self.standardize_data:
            # Request cache to return observations relative to current agent
            obs_frame: StateArray = convert_to_frame_state(
                raw_state,
                stationary=not standardize_derivatives,
                grounded=True,
            )
            cache.set_obs_frame(obs_frame)

            # Create and store 2d tranformation matrix to agent from world
            agent_pos = self.curr_agent_state_np.position
            agent_heading_vector = self.curr_agent_state_np.heading_vector
            cos_agent, sin_agent = agent_heading_vector[0], agent_heading_vector[1]
            world_from_agent_tf: np.ndarray = np.array(
                [
                    [cos_agent, -sin_agent, agent_pos[0]],
                    [sin_agent, cos_agent, agent_pos[1]],
                    [0.0, 0.0, 1.0],
                ]
            )
            self.agent_from_world_tf: np.ndarray = np.linalg.inv(world_from_agent_tf)
        else:
            self.agent_from_world_tf: np.ndarray = np.eye(3)

        ### AGENT-SPECIFIC DATA ###
        self.agent_history_np, self.agent_history_extent_np = self.get_agent_history(
            agent_info, history_sec
        )
        self.agent_history_len: int = self.agent_history_np.shape[0]

        self.agent_future_np, self.agent_future_extent_np = self.get_agent_future(
            agent_info, future_sec
        )
        self.agent_future_len: int = self.agent_future_np.shape[0]
        self.agent_meta_dict: Dict = get_agent_meta_dict(self.cache, agent_info)

        ### NEIGHBOR-SPECIFIC DATA ###
        def distance_limit(agent_types: np.ndarray, target_type: int) -> np.ndarray:
            return np.array(
                [
                    agent_interaction_distances[(agent_type, target_type)]
                    for agent_type in agent_types
                ]
            )

        nearby_agents, self.neighbor_types_np = self.get_nearby_agents(
            scene_time_agent, agent_info, distance_limit
        )

        self.num_neighbors = len(nearby_agents)
        (
            self.neighbor_histories,
            self.neighbor_history_extents,
            self.neighbor_history_lens_np,
        ) = self.get_neighbor_history(history_sec, nearby_agents)

        (
            self.neighbor_futures,
            self.neighbor_future_extents,
            self.neighbor_future_lens_np,
        ) = self.get_neighbor_future(future_sec, nearby_agents)

        self.neighbor_meta_dicts: Dict = [
            get_agent_meta_dict(self.cache, agent) for agent in nearby_agents
        ]

        ### ROBOT DATA ###
        self.robot_future_np: Optional[StateArray] = None

        if incl_robot_future:
            self.robot_future_np: StateArray = self.get_robot_current_and_future(
                scene_time_agent.robot, future_sec
            )

            # -1 because this is meant to hold the number of future steps
            # (whereas the above returns the current + future, yielding
            # one more timestep).
            self.robot_future_len: int = self.robot_future_np.shape[0] - 1

        ### MAP ###
        self.map_name: Optional[str] = None
        self.map_patch: Optional[RasterizedMapPatch] = None

        map_name: str = (
            f"{scene_time_agent.scene.env_name}:{scene_time_agent.scene.location}"
        )
        if incl_raster_map:
            self.map_name = map_name
            self.map_patch = self.get_agent_map_patch(raster_map_params)

        self.vec_map: Optional[VectorMap] = None
        if map_api is not None:
            self.vec_map = map_api.get_map(
                map_name,
                self.cache if self.cache.is_traffic_light_data_cached() else None,
                **vector_map_params if vector_map_params is not None else None,
            )

        self.scene_id = scene_time_agent.scene.name

        # Will be optionally populated by the user's provided functions.
        self.extras: Dict[str, np.ndarray] = dict()

    def get_agent_history(
        self,
        agent_info: AgentMetadata,
        history_sec: Tuple[Optional[float], Optional[float]],
    ) -> Tuple[StateArray, np.ndarray]:
        agent_history_np, agent_extent_history_np = self.cache.get_agent_history(
            agent_info, self.scene_ts, history_sec
        )
        return agent_history_np, agent_extent_history_np

    def get_agent_future(
        self,
        agent_info: AgentMetadata,
        future_sec: Tuple[Optional[float], Optional[float]],
    ) -> Tuple[StateArray, np.ndarray]:
        agent_future_np, agent_extent_future_np = self.cache.get_agent_future(
            agent_info, self.scene_ts, future_sec
        )
        return agent_future_np, agent_extent_future_np

    # @profile
    def get_nearby_agents(
        self,
        scene_time: SceneTimeAgent,
        agent_info: AgentMetadata,
        distance_limit: Callable[[np.ndarray, int], np.ndarray],
    ) -> Tuple[List[AgentMetadata], np.ndarray]:
        """
        Returns Agent Metadata and Agent types of nearby agents
        """
        # The indices of the returned ndarray match the scene_time agents list
        # (including the index of the central agent, which would have a distance
        # of 0 to itself).
        agent_distances: np.ndarray = scene_time.get_agent_distances_to(agent_info)
        agent_idx: int = scene_time.agents.index(agent_info)

        neighbor_types: np.ndarray = np.array([a.type.value for a in scene_time.agents])
        nearby_mask: np.ndarray = agent_distances <= distance_limit(
            neighbor_types, agent_info.type
        )
        nearby_mask[agent_idx] = False

        nb_idx = agent_distances.argsort()
        nearby_agents: List[AgentMetadata] = [
            scene_time.agents[idx] for idx in nb_idx if nearby_mask[idx]
        ]

        if self.max_neighbor_num is not None:
            # Pruning nearby_agents and re-creating
            # neighbor_types_np with the remaining agents.
            nearby_agents = nearby_agents[: self.max_neighbor_num]

        # Doing this here because the argsort above changes the order of agents.
        neighbor_types_np: np.ndarray = np.array([a.type.value for a in nearby_agents])

        return nearby_agents, neighbor_types_np

    def get_neighbor_history(
        self,
        history_sec: Tuple[Optional[float], Optional[float]],
        nearby_agents: List[AgentMetadata],
    ) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
        (
            neighbor_data,
            neighbor_extents_data,
            neighbor_data_lens_np,
        ) = self.cache.get_agents_history(self.scene_ts, nearby_agents, history_sec)
        return (
            neighbor_data,
            neighbor_extents_data,
            neighbor_data_lens_np,
        )

    def get_neighbor_future(
        self,
        future_sec: Tuple[Optional[float], Optional[float]],
        nearby_agents: List[AgentMetadata],
    ) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
        (
            neighbor_data,
            neighbor_extents_data,
            neighbor_data_lens_np,
        ) = self.cache.get_agents_future(self.scene_ts, nearby_agents, future_sec)
        return (
            neighbor_data,
            neighbor_extents_data,
            neighbor_data_lens_np,
        )

    def get_robot_current_and_future(
        self,
        robot_info: AgentMetadata,
        future_sec: Tuple[Optional[float], Optional[float]],
    ) -> np.ndarray:
        robot_curr_np: np.ndarray = self.cache.get_state(robot_info.name, self.scene_ts)
        # robot_fut_extents_np,
        (
            robot_fut_np,
            _,
        ) = self.cache.get_agent_future(robot_info, self.scene_ts, future_sec)

        robot_curr_and_fut_np: np.ndarray = np.concatenate(
            (robot_curr_np[np.newaxis, :], robot_fut_np), axis=0
        )
        return robot_curr_and_fut_np

    def get_agent_map_patch(self, patch_params: Dict[str, int]) -> RasterizedMapPatch:
        world_x, world_y = self.curr_agent_state_np.position
        desired_patch_size: int = patch_params["map_size_px"]
        resolution: float = patch_params["px_per_m"]
        offset_xy: Tuple[float, float] = patch_params.get("offset_frac_xy", (0.0, 0.0))
        return_rgb: bool = patch_params.get("return_rgb", True)
        no_map_fill_val: float = patch_params.get("no_map_fill_value", 0.0)

        if self.standardize_data:
            heading = self.curr_agent_state_np.heading[0]
            patch_data, raster_from_world_tf, has_data = self.cache.load_map_patch(
                world_x,
                world_y,
                desired_patch_size,
                resolution,
                offset_xy,
                heading,
                return_rgb,
                rot_pad_factor=sqrt(2),
                no_map_val=no_map_fill_val,
            )
        else:
            heading = 0.0
            patch_data, raster_from_world_tf, has_data = self.cache.load_map_patch(
                world_x,
                world_y,
                desired_patch_size,
                resolution,
                offset_xy,
                heading,
                return_rgb,
                no_map_val=no_map_fill_val,
            )

        return RasterizedMapPatch(
            data=patch_data,
            rot_angle=heading,
            crop_size=desired_patch_size,
            resolution=resolution,
            raster_from_world_tf=raster_from_world_tf,
            has_data=has_data,
        )


class SceneBatchElement:
    """A single batch element."""

    def __init__(
        self,
        cache: SceneCache,
        data_index: int,
        scene_time: SceneTime,
        history_sec: Tuple[Optional[float], Optional[float]],
        future_sec: Tuple[Optional[float], Optional[float]],
        agent_interaction_distances: Dict[
            Tuple[AgentType, AgentType], float
        ] = defaultdict(lambda: np.inf),
        incl_robot_future: bool = False,
        incl_raster_map: bool = False,
        raster_map_params: Optional[Dict[str, Any]] = None,
        map_api: Optional[MapAPI] = None,
        vector_map_params: Optional[Dict[str, Any]] = None,
        state_format: Optional[str] = None,
        standardize_data: bool = False,
        standardize_derivatives: bool = False,
        max_agent_num: Optional[int] = None,
    ) -> None:
        self.cache: SceneCache = cache
        self.data_index = data_index
        self.dt: float = scene_time.scene.dt
        self.scene_ts: int = scene_time.ts

        if max_agent_num is not None:
            scene_time.agents = scene_time.agents[:max_agent_num]

        self.agents: List[AgentMetadata] = scene_time.agents

        robot = [agent for agent in self.agents if agent.name == "ego"]
        if len(robot) > 0:
            self.centered_agent = robot[0]
        else:
            self.centered_agent = self.agents[0]

        raw_state: StateArray = cache.get_raw_state(
            self.centered_agent.name, self.scene_ts
        )

        if state_format is not None:
            self.centered_agent_state_np = raw_state.as_format(state_format)
        else:
            self.centered_agent_state_np = raw_state

        self.standardize_data = standardize_data

        if self.standardize_data:
            # Request cache to return observations relative to centered agent
            obs_frame: StateArray = convert_to_frame_state(
                raw_state,
                stationary=not standardize_derivatives,
                grounded=True,
            )
            cache.set_obs_frame(obs_frame)

            # Create 2d transformation matrix to and from agent and world
            agent_pos: np.ndarray = self.centered_agent_state_np.position
            agent_heading: float = self.centered_agent_state_np.heading[0]

            cos_agent, sin_agent = np.cos(agent_heading), np.sin(agent_heading)
            self.centered_world_from_agent_tf: np.ndarray = np.array(
                [
                    [cos_agent, -sin_agent, agent_pos[0]],
                    [sin_agent, cos_agent, agent_pos[1]],
                    [0.0, 0.0, 1.0],
                ]
            )
            self.centered_agent_from_world_tf: np.ndarray = np.linalg.inv(
                self.centered_world_from_agent_tf
            )
        else:
            self.centered_agent_from_world_tf: np.ndarray = np.eye(3)
            self.centered_world_from_agent_tf: np.ndarray = np.eye(3)

        ### NEIGHBOR-SPECIFIC DATA ###
        def distance_limit(agent_types: np.ndarray, target_type: int) -> np.ndarray:
            return np.array(
                [
                    agent_interaction_distances[(agent_type, target_type)]
                    for agent_type in agent_types
                ]
            )

        nearby_agents, self.agent_types_np = self.get_nearby_agents(
            scene_time, self.centered_agent, distance_limit
        )

        self.num_agents = len(nearby_agents)
        self.agent_names = [agent.name for agent in nearby_agents]
        (
            self.agent_histories,
            self.agent_history_extents,
            self.agent_history_lens_np,
        ) = self.get_agents_history(history_sec, nearby_agents)
        (
            self.agent_futures,
            self.agent_future_extents,
            self.agent_future_lens_np,
        ) = self.get_agents_future(future_sec, nearby_agents)

        self.agent_meta_dicts = [
            get_agent_meta_dict(self.cache, agent) for agent in nearby_agents
        ]

        ### MAP ###
        self.map_name: Optional[str] = None
        self.map_patches: Optional[RasterizedMapPatch] = None

        map_name: str = f"{scene_time.scene.env_name}:{scene_time.scene.location}"
        if incl_raster_map:
            self.map_name = map_name
            self.map_patches = self.get_agents_map_patch(
                raster_map_params, self.agent_histories
            )

        self.vec_map: Optional[VectorMap] = None
        if map_api is not None:
            self.vec_map = map_api.get_map(
                map_name,
                self.cache if self.cache.is_traffic_light_data_cached() else None,
                **vector_map_params if vector_map_params is not None else None,
            )

        self.scene_id = scene_time.scene.name

        ### ROBOT DATA ###
        self.robot_future_np: Optional[StateArray] = None

        if incl_robot_future:
            self.robot_future_np: StateArray = self.get_robot_current_and_future(
                self.centered_agent, future_sec
            )

            # -1 because this is meant to hold the number of future steps
            # (whereas the above returns the current + future, yielding
            # one more timestep).
            self.robot_future_len: int = self.robot_future_np.shape[0] - 1

        self.scene_id = scene_time.scene.name

        # Will be optionally populated by the user's provided functions.
        self.extras: Dict[str, np.ndarray] = dict()

    def get_nearby_agents(
        self,
        scene_time: SceneTime,
        agent: AgentMetadata,
        distance_limit: Callable[[np.ndarray, int], np.ndarray],
    ) -> Tuple[List[AgentMetadata], np.ndarray]:
        agent_distances: np.ndarray = scene_time.get_agent_distances_to(agent)

        agents_types: np.ndarray = np.array([a.type.value for a in scene_time.agents])
        nearby_mask: np.ndarray = agent_distances <= distance_limit(
            agents_types, agent.type
        )
        # sort the agents based on their distance to the centered agent
        idx = np.argsort(agent_distances)
        num_qualified = nearby_mask.sum()
        nearby_agents: List[AgentMetadata] = [
            scene_time.agents[idx[i]] for i in range(num_qualified)
        ]
        agents_types_np = agents_types[idx[:num_qualified]]
        return nearby_agents, agents_types_np

    def get_agents_history(
        self,
        history_sec: Tuple[Optional[float], Optional[float]],
        nearby_agents: List[AgentMetadata],
    ) -> Tuple[List[StateArray], List[np.ndarray], np.ndarray]:
        # The indices of the returned ndarray match the scene_time agents list (including the index of the central agent,
        # which would have a distance of 0 to itself).
        (
            agent_histories,
            agent_history_extents,
            agent_history_lens_np,
        ) = self.cache.get_agents_history(self.scene_ts, nearby_agents, history_sec)

        return (
            agent_histories,
            agent_history_extents,
            agent_history_lens_np,
        )

    def get_agents_future(
        self,
        future_sec: Tuple[Optional[float], Optional[float]],
        nearby_agents: List[AgentMetadata],
    ) -> Tuple[List[StateArray], List[np.ndarray], np.ndarray]:

        (
            agent_futures,
            agent_future_extents,
            agent_future_lens_np,
        ) = self.cache.get_agents_future(self.scene_ts, nearby_agents, future_sec)

        return (
            agent_futures,
            agent_future_extents,
            agent_future_lens_np,
        )

    def get_agents_map_patch(
        self, patch_params: Dict[str, int], agent_histories: List[np.ndarray]
    ) -> List[RasterizedMapPatch]:
        desired_patch_size: int = patch_params["map_size_px"]
        resolution: float = patch_params["px_per_m"]
        offset_xy: Tuple[float, float] = patch_params.get("offset_frac_xy", (0.0, 0.0))
        return_rgb: bool = patch_params.get("return_rgb", True)
        no_map_fill_val: float = patch_params.get("no_map_fill_value", 0.0)

        map_patches = list()

        curr_state = [state[-1] for state in agent_histories]
        curr_state = np.stack(curr_state).view(self.cache.obs_type)

        if self.standardize_data:
            # need to transform back into world frame
            obs_frame: StateArray = convert_to_frame_state(
                self.centered_agent_state_np, stationary=True, grounded=True
            )
            curr_state = transform_from_frame(curr_state, obs_frame)
            heading = curr_state.heading[:, 0]
        else:
            heading = 0.0 * curr_state.heading[:, 0]

        for i in range(curr_state.shape[0]):
            patch_data, raster_from_world_tf, has_data = self.cache.load_map_patch(
                curr_state.get_attr("x")[i],
                curr_state.get_attr("y")[i],
                desired_patch_size,
                resolution,
                offset_xy,
                heading[i],
                return_rgb,
                no_map_val=no_map_fill_val,
            )
            map_patches.append(
                RasterizedMapPatch(
                    data=patch_data,
                    rot_angle=heading[i],
                    crop_size=desired_patch_size,
                    resolution=resolution,
                    raster_from_world_tf=raster_from_world_tf,
                    has_data=has_data,
                )
            )

        return map_patches

    def get_robot_current_and_future(
        self,
        robot_info: AgentMetadata,
        future_sec: Tuple[Optional[float], Optional[float]],
    ) -> StateArray:
        robot_curr_np: StateArray = self.cache.get_state(robot_info.name, self.scene_ts)
        # robot_fut_extents_np,
        (
            robot_fut_np,
            _,
        ) = self.cache.get_agent_future(robot_info, self.scene_ts, future_sec)

        robot_curr_and_fut_np: StateArray = np.concatenate(
            (robot_curr_np[np.newaxis, :], robot_fut_np), axis=0
        ).view(self.cache.obs_type)
        return robot_curr_and_fut_np


def is_agent_stationary(cache: SceneCache, agent_info: AgentMetadata) -> bool:
    # Agent is considered stationary if it moves less than 1m between the first and last valid timestep.
    first_state: StateArray = cache.get_state(
        agent_info.name, agent_info.first_timestep
    )
    last_state: StateArray = cache.get_state(agent_info.name, agent_info.last_timestep)
    is_stationary = np.square(last_state.position - first_state.position).sum(0) < 1.0
    return is_stationary


def get_agent_meta_dict(
    cache: SceneCache, agent_info: AgentMetadata
) -> Dict[str, np.ndarray]:
    return {
        "is_stationary": is_agent_stationary(cache, agent_info),
    }
