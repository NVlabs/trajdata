from collections import defaultdict
from math import sqrt
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from trajdata.caching import SceneCache
from trajdata.data_structures.agent import Agent, AgentMetadata, AgentType, FixedExtent
from trajdata.data_structures.map_patch import MapPatch
from trajdata.data_structures.scene import SceneTime, SceneTimeAgent


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
        incl_map: bool = False,
        map_params: Optional[Dict[str, int]] = None,
        standardize_data: bool = False,
        standardize_derivatives: bool = False,
    ) -> None:
        self.cache: SceneCache = cache
        self.data_index: int = data_index
        self.dt: float = scene_time_agent.scene.dt
        self.scene_ts: int = scene_time_agent.ts

        agent_info: AgentMetadata = scene_time_agent.agent
        self.agent_name: str = agent_info.name
        self.agent_type: AgentType = agent_info.type

        self.curr_agent_state_np: np.ndarray = cache.get_state(
            agent_info.name, self.scene_ts
        )

        self.standardize_data = standardize_data
        if self.standardize_data:
            agent_pos: np.ndarray = self.curr_agent_state_np[:2]
            agent_heading: float = self.curr_agent_state_np[-1]

            cos_agent, sin_agent = np.cos(agent_heading), np.sin(agent_heading)
            world_from_agent_tf: np.ndarray = np.array(
                [
                    [cos_agent, -sin_agent, agent_pos[0]],
                    [sin_agent, cos_agent, agent_pos[1]],
                    [0.0, 0.0, 1.0],
                ]
            )
            self.agent_from_world_tf: np.ndarray = np.linalg.inv(world_from_agent_tf)

            offset = self.curr_agent_state_np
            if not standardize_derivatives:
                offset[2:6] = 0.0

            cache.transform_data(
                shift_mean_to=offset,
                rotate_by=agent_heading,
                sincos_heading=True,
            )

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

        ### NEIGHBOR-SPECIFIC DATA ###
        def distance_limit(agent_types: np.ndarray, target_type: int) -> np.ndarray:
            return np.array(
                [
                    agent_interaction_distances[(agent_type, target_type)]
                    for agent_type in agent_types
                ]
            )

        (
            self.num_neighbors,
            self.neighbor_types_np,
            self.neighbor_histories,
            self.neighbor_history_extents,
            self.neighbor_history_lens_np,
        ) = self.get_neighbor_history(
            scene_time_agent, agent_info, history_sec, distance_limit
        )

        (
            _,
            _,
            self.neighbor_futures,
            self.neighbor_future_extents,
            self.neighbor_future_lens_np,
        ) = self.get_neighbor_future(
            scene_time_agent, agent_info, future_sec, distance_limit
        )

        ### ROBOT DATA ###
        self.robot_future_np: Optional[np.ndarray] = None

        if incl_robot_future:
            self.robot_future_np: np.ndarray = self.get_robot_current_and_future(
                scene_time_agent.robot, future_sec
            )

            # -1 because this is meant to hold the number of future steps
            # (whereas the above returns the current + future, yielding
            # one more timestep).
            self.robot_future_len: int = self.robot_future_np.shape[0] - 1

        ### MAP ###
        self.map_patch: Optional[MapPatch] = None
        if incl_map:
            self.map_patch = self.get_agent_map_patch(map_params)

        self.scene_id = scene_time_agent.scene.name

    def get_agent_history(
        self,
        agent_info: AgentMetadata,
        history_sec: Tuple[Optional[float], Optional[float]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        agent_history_np, agent_extent_history_np = self.cache.get_agent_history(
            agent_info, self.scene_ts, history_sec
        )
        return agent_history_np, agent_extent_history_np

    def get_agent_future(
        self,
        agent_info: AgentMetadata,
        future_sec: Tuple[Optional[float], Optional[float]],
    ) -> np.ndarray:
        agent_future_np, agent_extent_future_np = self.cache.get_agent_future(
            agent_info, self.scene_ts, future_sec
        )
        return agent_future_np, agent_extent_future_np

    def get_neighbor_history(
        self,
        scene_time: SceneTimeAgent,
        agent_info: AgentMetadata,
        history_sec: Tuple[Optional[float], Optional[float]],
        distance_limit: Callable[[np.ndarray, int], np.ndarray],
    ) -> Tuple[int, np.ndarray, List[np.ndarray], List[np.ndarray], np.ndarray]:
        # The indices of the returned ndarray match the scene_time agents list (including the index of the central agent,
        # which would have a distance of 0 to itself).
        agent_distances: np.ndarray = scene_time.get_agent_distances_to(agent_info)
        agent_idx: int = scene_time.agents.index(agent_info)
        neighbor_types: np.ndarray = np.array([a.type.value for a in scene_time.agents])
        nearby_mask: np.ndarray = agent_distances <= distance_limit(
            neighbor_types, agent_info.type
        )
        nearby_mask[agent_idx] = False

        nearby_agents: List[AgentMetadata] = [
            agent for (idx, agent) in enumerate(scene_time.agents) if nearby_mask[idx]
        ]
        neighbor_types_np: np.ndarray = neighbor_types[nearby_mask]

        num_neighbors: int = len(nearby_agents)
        (
            neighbor_histories,
            neighbor_history_extents,
            neighbor_history_lens_np,
        ) = self.cache.get_agents_history(self.scene_ts, nearby_agents, history_sec)

        return (
            num_neighbors,
            neighbor_types_np,
            neighbor_histories,
            neighbor_history_extents,
            neighbor_history_lens_np,
        )

    # @profile
    def get_neighbor_future(
        self,
        scene_time: SceneTimeAgent,
        agent_info: AgentMetadata,
        future_sec: Tuple[Optional[float], Optional[float]],
        distance_limit: Callable[[np.ndarray, int], np.ndarray],
    ) -> Tuple[int, np.ndarray, List[np.ndarray], List[np.ndarray], np.ndarray]:
        scene_ts: int = self.scene_ts

        # The indices of the returned ndarray match the scene_time agents list (including the index of the central agent,
        # which would have a distance of 0 to itself).
        agent_distances: np.ndarray = scene_time.get_agent_distances_to(agent_info)
        agent_idx: int = scene_time.agents.index(agent_info)

        neighbor_types: np.ndarray = np.array([a.type.value for a in scene_time.agents])
        nearby_mask: np.ndarray = agent_distances <= distance_limit(
            neighbor_types, agent_info.type
        )
        nearby_mask[agent_idx] = False

        nearby_agents: List[AgentMetadata] = [
            agent for (idx, agent) in enumerate(scene_time.agents) if nearby_mask[idx]
        ]
        neighbor_types_np: np.ndarray = neighbor_types[nearby_mask]

        num_neighbors: int = len(nearby_agents)
        (
            neighbor_futures,
            neighbor_future_extents,
            neighbor_future_lens_np,
        ) = self.cache.get_agents_future(scene_ts, nearby_agents, future_sec)

        return (
            num_neighbors,
            neighbor_types_np,
            neighbor_futures,
            neighbor_future_extents,
            neighbor_future_lens_np,
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

    def get_agent_map_patch(self, patch_params: Dict[str, int]) -> MapPatch:
        world_x, world_y = self.curr_agent_state_np[:2]
        desired_patch_size: int = patch_params["map_size_px"]
        resolution: int = patch_params["px_per_m"]
        offset_xy: Tuple[float, float] = patch_params.get("offset_frac_xy", (0.0, 0.0))
        return_rgb: bool = patch_params.get("return_rgb", True)

        if self.standardize_data:
            heading = self.curr_agent_state_np[-1]
            patch_data, raster_from_world_tf = self.cache.load_map_patch(
                world_x,
                world_y,
                desired_patch_size,
                resolution,
                offset_xy,
                heading,
                return_rgb,
                rot_pad_factor=sqrt(2),
            )
        else:
            heading = 0.0
            patch_data, raster_from_world_tf = self.cache.load_map_patch(
                world_x,
                world_y,
                desired_patch_size,
                resolution,
                offset_xy,
                heading,
                return_rgb,
            )

        return MapPatch(
            data=patch_data,
            rot_angle=heading,
            crop_size=desired_patch_size,
            resolution=resolution,
            raster_from_world_tf=raster_from_world_tf,
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
        incl_map: bool = False,
        map_params: Optional[Dict[str, int]] = None,
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

        self.centered_agent_state_np: np.ndarray = cache.get_state(
            self.centered_agent.name, self.scene_ts
        )
        self.standardize_data = standardize_data

        if self.standardize_data:
            agent_pos: np.ndarray = self.centered_agent_state_np[:2]
            agent_heading: float = self.centered_agent_state_np[-1]

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

            offset = self.centered_agent_state_np
            if not standardize_derivatives:
                offset[2:6] = 0.0

            cache.transform_data(
                shift_mean_to=offset,
                rotate_by=agent_heading,
                sincos_heading=True,
            )
        else:
            self.agent_from_world_tf: np.ndarray = np.eye(3)

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

        ### MAP ###
        self.map_patches: Optional[MapPatch] = None
        if incl_map:
            self.map_patches = self.get_agents_map_patch(
                map_params, self.agent_histories
            )

        ### ROBOT DATA ###
        self.robot_future_np: Optional[np.ndarray] = None

        if incl_robot_future:
            self.robot_future_np: np.ndarray = self.get_robot_current_and_future(
                self.centered_agent, future_sec
            )

            # -1 because this is meant to hold the number of future steps
            # (whereas the above returns the current + future, yielding
            # one more timestep).
            self.robot_future_len: int = self.robot_future_np.shape[0] - 1

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
    ) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
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
    ) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:

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
    ) -> List[MapPatch]:
        world_x, world_y = self.centered_agent_state_np[:2]
        heading = self.centered_agent_state_np[-1]
        desired_patch_size: int = patch_params["map_size_px"]
        resolution: int = patch_params["px_per_m"]
        offset_xy: Tuple[float, float] = patch_params.get("offset_frac_xy", (0.0, 0.0))
        return_rgb: bool = patch_params.get("return_rgb", True)

        if len(self.cache.heading_cols) == 1:
            heading_idx = self.cache.heading_cols[0]
            sincos = False
        else:
            heading_sin_idx, heading_cos_idx = self.cache.heading_cols
            sincos = True
        x_idx, y_idx = self.cache.pos_cols

        map_patches = list()
        for agent_his in agent_histories:
            if self.standardize_data:
                if sincos:
                    agent_heading = (
                        np.arctan2(
                            agent_his[-1, heading_sin_idx],
                            agent_his[-1, heading_cos_idx],
                        )
                        + heading
                    )
                else:
                    agent_heading = agent_his[-1, heading_idx] + heading

                patch_data, raster_from_world_tf = self.cache.load_map_patch(
                    world_x + agent_his[-1, x_idx],
                    world_y + agent_his[-1, y_idx],
                    desired_patch_size,
                    resolution,
                    offset_xy,
                    agent_heading,
                    return_rgb,
                    rot_pad_factor=sqrt(2),
                )

            else:
                agent_heading = 0.0
                patch_data, raster_from_world_tf = self.cache.load_map_patch(
                    agent_his[-1, x_idx],
                    agent_his[-1, y_idx],
                    desired_patch_size,
                    resolution,
                    offset_xy,
                    agent_heading,
                    return_rgb,
                )

            map_patches.append(
                MapPatch(
                    data=patch_data,
                    rot_angle=agent_heading,
                    crop_size=desired_patch_size,
                    resolution=resolution,
                    raster_from_world_tf=raster_from_world_tf,
                )
            )

        return map_patches

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
