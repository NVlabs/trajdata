from collections import defaultdict
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
from torch.utils.data import Sampler

from trajdata import UnifiedDataset
from trajdata.data_structures import (
    AgentBatch,
    AgentBatchElement,
    AgentDataIndex,
    AgentType,
    SceneBatchElement,
    SceneTimeAgent,
)
from trajdata.data_structures.collation import agent_collate_fn


class SceneTimeBatcher(Sampler):
    _agent_data_index: AgentDataIndex
    _agent_idx: int

    def __init__(
        self, agent_centric_dataset: UnifiedDataset, agent_idx_to_follow: int = 0
    ) -> None:
        """
        Returns a sampler (to be used in a torch.utils.data.DataLoader)
        which works with an agent-centric UnifiedDataset, yielding
        batches consisting of whole scenes (AgentBatchElements for all agents
        in a particular scene at a particular time)

        Args:
            agent_centric_dataset (UnifiedDataset)
            agent_idx_to_follow (int): index of agent to return batches for. Defaults to 0,
                meaning we include all scene frames where the ego agent appears, which
                usually covers the entire dataset.
        """
        super().__init__(agent_centric_dataset)
        self._agent_data_index = agent_centric_dataset._data_index
        self._agent_idx = agent_idx_to_follow
        self._cumulative_lengths = np.concatenate(
            [
                [0],
                np.cumsum(
                    [
                        cumulative_scene_length[self._agent_idx + 1]
                        - cumulative_scene_length[self._agent_idx]
                        for cumulative_scene_length in self._agent_data_index._cumulative_scene_lengths
                    ]
                ),
            ]
        )

    def __len__(self):
        return self._cumulative_lengths[-1]

    def __iter__(self) -> Iterator[int]:
        for idx in range(len(self)):
            # TODO(apoorvas) May not need to do this search, since we only support an iterable style access?
            scene_idx: int = (
                np.searchsorted(self._cumulative_lengths, idx, side="right").item() - 1
            )

            # offset into dataset index to reach current scene
            scene_offset = self._agent_data_index._cumulative_lengths[scene_idx].item()

            # how far along we are in the current scene
            scene_elem_index = idx - self._cumulative_lengths[scene_idx].item()

            # convert to scene-timestep for the tracked agent
            scene_ts = (
                scene_elem_index
                + self._agent_data_index._agent_times[scene_idx][self._agent_idx, 0]
            )

            # build a set of indices into the agent-centric dataset for all agents that exist at this scene and timestep
            indices = []
            for agent_idx, agent_times in enumerate(
                self._agent_data_index._agent_times[scene_idx]
            ):
                if scene_ts > agent_times[1]:
                    # we are past the last timestep for this agent (times are inclusive)
                    continue
                agent_offset = scene_ts - agent_times[0]
                if agent_offset < 0:
                    # this agent hasn't entered the scene yet
                    continue

                # compute index into original dataset, first into scene, then into this agent's part in scene, and then the offset
                index_to_add = (
                    scene_offset
                    + self._agent_data_index._cumulative_scene_lengths[scene_idx][
                        agent_idx
                    ]
                    + agent_offset
                )
                indices.append(index_to_add)

            yield indices


def convert_to_agent_batch(
    scene_batch_element: SceneBatchElement,
    only_types: Optional[List[AgentType]] = None,
    no_types: Optional[List[AgentType]] = None,
    agent_interaction_distances: Dict[Tuple[AgentType, AgentType], float] = defaultdict(
        lambda: np.inf
    ),
    incl_map: bool = False,
    map_params: Optional[Dict[str, Any]] = None,
    max_neighbor_num: Optional[int] = None,
    state_format: Optional[str] = None,
    standardize_data: bool = True,
    standardize_derivatives: bool = False,
    pad_format: str = "outside",
) -> AgentBatch:
    """
    Converts a SceneBatchElement into a AgentBatch consisting of
    AgentBatchElements for all agents present at the given scene at the given
    time step.

    Args:
        scene_batch_element (SceneBatchElement): element to process
        only_types (Optional[List[AgentType]], optional): AgentsTypes to consider. Defaults to None.
        no_types (Optional[List[AgentType]], optional): AgentTypes to ignore. Defaults to None.
        agent_interaction_distances (_type_, optional): Distance threshold for interaction. Defaults to defaultdict(lambda: np.inf).
        incl_map (bool, optional): Whether to include map info. Defaults to False.
        map_params (Optional[Dict[str, Any]], optional): Map params. Defaults to None.
        max_neighbor_num (Optional[int], optional): Max number of neighbors to allow. Defaults to None.
        standardize_data (bool): Whether to return data relative to current agent state. Defaults to True.
        standardize_derivatives: Whether to transform relative velocities and accelerations as well. Defaults to False.
        pad_format (str, optional): Pad format when collating agent trajectories. Defaults to "outside".

    Returns:
        AgentBatch: batch of AgentBatchElements corresponding to all agents in the SceneBatchElement
    """
    data_idx = scene_batch_element.data_index
    cache = scene_batch_element.cache
    scene = cache.scene
    dt = scene_batch_element.dt
    ts = scene_batch_element.scene_ts
    state_format = scene_batch_element.centered_agent_state_np._format

    batch_elems: List[AgentBatchElement] = []
    for j, agent_name in enumerate(scene_batch_element.agent_names):
        history_sec = dt * (scene_batch_element.agent_histories[j].shape[0] - 1)
        future_sec = dt * (scene_batch_element.agent_futures[j].shape[0])
        cache.reset_obs_frame()
        scene_time_agent: SceneTimeAgent = SceneTimeAgent.from_cache(
            scene,
            ts,
            agent_name,
            cache,
            only_types=only_types,
            no_types=no_types,
        )

        batch_elems.append(
            AgentBatchElement(
                cache=cache,
                data_index=data_idx,
                scene_time_agent=scene_time_agent,
                history_sec=(history_sec, history_sec),
                future_sec=(future_sec, future_sec),
                agent_interaction_distances=agent_interaction_distances,
                incl_raster_map=incl_map,
                raster_map_params=map_params,
                state_format=state_format,
                standardize_data=standardize_data,
                standardize_derivatives=standardize_derivatives,
                max_neighbor_num=max_neighbor_num,
            )
        )

    return agent_collate_fn(batch_elems, return_dict=False, pad_format=pad_format)
