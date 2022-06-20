from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from trajdata import filtering
from trajdata.augmentation import BatchAugmentation
from trajdata.caching.df_cache import DataFrameCache
from trajdata.data_structures.agent import AgentMetadata, VariableExtent
from trajdata.data_structures.batch import AgentBatch
from trajdata.data_structures.batch_element import AgentBatchElement
from trajdata.data_structures.collation import agent_collate_fn
from trajdata.data_structures.scene import SceneTimeAgent
from trajdata.data_structures.scene_metadata import Scene
from trajdata.dataset import UnifiedDataset
from trajdata.simulation.sim_cache import SimulationCache
from trajdata.simulation.sim_df_cache import SimulationDataFrameCache
from trajdata.simulation.sim_metrics import SimMetric
from trajdata.simulation.sim_stats import SimStatistic


class SimulationScene:
    def __init__(
        self,
        env_name: str,
        scene_name: str,
        scene: Scene,
        dataset: UnifiedDataset,
        init_timestep: int = 0,
        freeze_agents: bool = True,
        return_dict: bool = False,
    ) -> None:
        self.env_name: str = env_name
        self.scene_name: str = scene_name
        self.scene_info: Scene = deepcopy(scene)
        self.dataset: UnifiedDataset = dataset
        self.init_scene_ts: int = init_timestep
        self.freeze_agents: bool = freeze_agents
        self.return_dict: bool = return_dict

        self.scene_ts: int = self.init_scene_ts

        agents_present: List[AgentMetadata] = self.scene_info.agent_presence[
            self.scene_ts
        ]
        self.agents: List[AgentMetadata] = filtering.agent_types(
            agents_present, self.dataset.no_types, self.dataset.only_types
        )

        if self.freeze_agents:
            self.scene_info.agent_presence = self.scene_info.agent_presence[
                : self.init_scene_ts + 1
            ]
            self.scene_info.agents = self.agents

        # Note this order of operations is important, we first instantiate
        # the cache with the copied scene_info + modified agents list.
        # Then, we change the env_name and etc later during finalization
        # (if we did it earlier then the cache would go looking inside
        # the sim folder for scene data rather than the original scene
        # data location).
        if self.dataset.cache_class == DataFrameCache:
            self.cache: SimulationCache = SimulationDataFrameCache(
                dataset.cache_path,
                self.scene_info,
                init_timestep,
                dataset.augmentations,
            )

        self.batch_augments: Optional[List[BatchAugmentation]] = None
        if dataset.augmentations:
            self.batch_augments = [
                batch_aug
                for batch_aug in dataset.augmentations
                if isinstance(batch_aug, BatchAugmentation)
            ]

    def reset(self) -> Union[AgentBatch, Dict[str, Any]]:
        self.scene_ts: int = self.init_scene_ts
        return self.get_obs()

    def step(
        self,
        new_xyh_dict: Dict[str, np.ndarray],
        return_obs=True,
    ) -> Union[AgentBatch, Dict[str, Any]]:
        self.scene_ts += 1

        self.cache.append_state(new_xyh_dict)

        if not self.freeze_agents:
            agents_present: List[AgentMetadata] = self.scene_info.agent_presence[
                self.scene_ts
            ]
            self.agents: List[AgentMetadata] = filtering.agent_types(
                agents_present, self.dataset.no_types, self.dataset.only_types
            )

            self.scene_info.agent_presence[self.scene_ts] = self.agents
        else:
            self.scene_info.agent_presence.append(self.agents)

        if return_obs:
            return self.get_obs()

    def get_obs(
        self, collate: bool = True, get_map: bool = True
    ) -> Union[AgentBatch, Dict[str, Any]]:
        agent_data_list: List[AgentBatchElement] = list()
        for agent in self.agents:
            scene_time_agent = SceneTimeAgent(
                self.scene_info, self.scene_ts, self.agents, agent, self.cache
            )

            agent_data_list.append(
                AgentBatchElement(
                    self.cache,
                    -1,  # Not used
                    scene_time_agent,
                    history_sec=self.dataset.history_sec,
                    future_sec=self.dataset.future_sec,
                    agent_interaction_distances=self.dataset.agent_interaction_distances,
                    incl_robot_future=False,
                    incl_map=get_map and self.dataset.incl_map,
                    map_params=self.dataset.map_params,
                    standardize_data=self.dataset.standardize_data,
                )
            )

            # Need to do reset for each agent since each
            # AgentBatchElement transforms (standardizes) the cache.
            self.cache.reset()

        if collate:
            return agent_collate_fn(
                agent_data_list,
                return_dict=self.return_dict,
                batch_augments=self.batch_augments,
            )
        else:
            return agent_data_list

    def get_metrics(self, metrics: List[SimMetric]) -> Dict[str, Dict[str, float]]:
        return self.cache.calculate_metrics(
            metrics, ts_range=(self.init_scene_ts + 1, self.scene_ts)
        )

    def get_stats(
        self, stats: List[SimStatistic]
    ) -> Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
        return self.cache.calculate_stats(
            stats, ts_range=(self.init_scene_ts + 1, self.scene_ts)
        )

    def finalize(self) -> None:
        # We only change the agent's last timestep here because we use it
        # earlier to check if the agent has any future data from the original
        # dataset.
        for agent in self.agents:
            agent.last_timestep = self.scene_ts

        self.scene_info.length_timesteps = self.scene_ts + 1

        self.scene_info.agent_presence = self.scene_info.agent_presence[
            : self.scene_ts + 1
        ]

        self.scene_info.env_metadata.name = self.env_name
        self.scene_info.env_name = self.env_name
        self.scene_info.name = self.scene_name

    def save(self) -> None:
        self.dataset.env_cache.save_scene(self.scene_info)
        self.cache.save_sim_scene(self.scene_info)
