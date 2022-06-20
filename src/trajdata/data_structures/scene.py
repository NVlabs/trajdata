from typing import List, Optional, Set

import numpy as np
import pandas as pd

from trajdata import filtering
from trajdata.caching import SceneCache
from trajdata.data_structures.agent import Agent, AgentMetadata, AgentType
from trajdata.data_structures.scene_metadata import Scene


class SceneTime:
    """Holds the data for a particular scene at a particular timestep."""

    def __init__(
        self,
        scene: Scene,
        scene_ts: int,
        agents: List[AgentMetadata],
        cache: SceneCache,
    ) -> None:
        self.scene = scene
        self.ts = scene_ts
        self.agents = agents
        self.cache = cache

    @classmethod
    def from_cache(
        cls,
        scene: Scene,
        scene_ts: int,
        cache: SceneCache,
        only_types: Optional[Set[AgentType]] = None,
        no_types: Optional[Set[AgentType]] = None,
    ):
        agents_present: List[AgentMetadata] = scene.agent_presence[scene_ts]
        filtered_agents: List[AgentMetadata] = filtering.agent_types(
            agents_present, no_types, only_types
        )

        data_df: pd.DataFrame = cache.load_all_agent_data(scene)

        agents: List[Agent] = list()
        for agent_info in filtered_agents:
            agents.append(Agent(agent_info, data_df.loc[agent_info.name]))

        return cls(scene, scene_ts, agents, cache)

    def get_agent_distances_to(self, agent: Agent) -> np.ndarray:
        agent_pos = np.array(
            [[agent.data.at[self.ts, "x"], agent.data.at[self.ts, "y"]]]
        )

        data_df: pd.DataFrame = self.cache.load_agent_xy_at_time(self.ts, self.scene)

        agent_ids = [a.name for a in self.agents]
        curr_poses = data_df.loc[agent_ids, ["x", "y"]].values
        return np.linalg.norm(curr_poses - agent_pos, axis=1)


class SceneTimeAgent:
    """Holds the data for a particular agent in a scene at a particular timestep."""

    def __init__(
        self,
        scene: Scene,
        scene_ts: int,
        agents: List[AgentMetadata],
        agent: AgentMetadata,
        cache: SceneCache,
        robot: Optional[AgentMetadata] = None,
    ) -> None:
        self.scene = scene
        self.ts = scene_ts
        self.agents = agents
        self.agent = agent
        self.cache = cache
        self.robot = robot

    @classmethod
    def from_cache(
        cls,
        scene: Scene,
        scene_ts: int,
        agent_id: str,
        cache: SceneCache,
        only_types: Optional[Set[AgentType]] = None,
        no_types: Optional[Set[AgentType]] = None,
        incl_robot_future: bool = False,
    ):
        agents_present: List[AgentMetadata] = scene.agent_presence[scene_ts]
        filtered_agents: List[AgentMetadata] = filtering.agent_types(
            agents_present, no_types, only_types
        )

        agent_metadata = next((a for a in filtered_agents if a.name == agent_id), None)

        if incl_robot_future:
            ego_metadata = next((a for a in filtered_agents if a.name == "ego"), None)

            return cls(
                scene,
                scene_ts,
                agents=filtered_agents,
                agent=agent_metadata,
                cache=cache,
                robot=ego_metadata,
            )
        else:
            return cls(
                scene,
                scene_ts,
                agents=filtered_agents,
                agent=agent_metadata,
                cache=cache,
            )

    # @profile
    def get_agent_distances_to(self, agent_info: AgentMetadata) -> np.ndarray:
        agent_pos = np.array(
            [
                [
                    self.cache.get_value(agent_info.name, self.ts, "x"),
                    self.cache.get_value(agent_info.name, self.ts, "y"),
                ]
            ]
        )

        curr_poses: np.ndarray = self.cache.get_positions_at(self.ts, self.agents)
        return np.linalg.norm(curr_poses - agent_pos, axis=1)
