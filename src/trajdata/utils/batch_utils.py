from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from trajdata.data_structures import (
    AgentBatch,
    AgentBatchElement,
    AgentType,
    SceneBatchElement,
    SceneTimeAgent,
)
from trajdata.data_structures.collation import agent_collate_fn


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
