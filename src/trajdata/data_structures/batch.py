from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch import Tensor

from trajdata.data_structures.agent import AgentType
from trajdata.maps import VectorMap
from trajdata.utils.arr_utils import PadDirection


@dataclass
class AgentBatch:
    data_idx: Tensor
    scene_ts: Tensor
    dt: Tensor
    agent_name: List[str]
    agent_type: Tensor
    curr_agent_state: Tensor
    agent_hist: Tensor
    agent_hist_extent: Tensor
    agent_hist_len: Tensor
    agent_fut: Tensor
    agent_fut_extent: Tensor
    agent_fut_len: Tensor
    num_neigh: Tensor
    neigh_types: Tensor
    neigh_hist: Tensor
    neigh_hist_extents: Tensor
    neigh_hist_len: Tensor
    neigh_fut: Tensor
    neigh_fut_extents: Tensor
    neigh_fut_len: Tensor
    robot_fut: Optional[Tensor]
    robot_fut_len: Optional[Tensor]
    map_names: Optional[List[str]]
    maps: Optional[Tensor]
    maps_resolution: Optional[Tensor]
    vector_maps: Optional[List[VectorMap]]
    rasters_from_world_tf: Optional[Tensor]
    agents_from_world_tf: Tensor
    scene_ids: Optional[List]
    history_pad_dir: PadDirection
    extras: Dict[str, Tensor]

    def to(self, device) -> None:
        excl_vals = {
            "data_idx",
            "agent_name",
            "agent_type",
            "agent_hist_len",
            "agent_fut_len",
            "neigh_hist_len",
            "neigh_fut_len",
            "neigh_types",
            "num_neigh",
            "robot_fut_len",
            "map_names",
            "vector_maps",
            "scene_ids",
            "history_pad_dir",
            "extras",
        }
        for val in vars(self).keys():
            tensor_val = getattr(self, val)
            if val not in excl_vals and tensor_val is not None:
                tensor_val: Tensor
                setattr(self, val, tensor_val.to(device, non_blocking=True))

        for key, val in self.extras.items():
            # Allow for custom .to() method for objects that define a __to__ function.
            if hasattr(val, "__to__"):
                self.extras[key] = val.__to__(device, non_blocking=True)
            else:
                self.extras[key] = val.to(device, non_blocking=True)

    def agent_types(self) -> List[AgentType]:
        unique_types: Tensor = torch.unique(self.agent_type)
        return [AgentType(unique_type.item()) for unique_type in unique_types]

    def for_agent_type(self, agent_type: AgentType) -> AgentBatch:
        match_type = self.agent_type == agent_type
        return AgentBatch(
            data_idx=self.data_idx[match_type],
            scene_ts=self.scene_ts[match_type],
            dt=self.dt[match_type],
            agent_name=[
                name for idx, name in enumerate(self.agent_name) if match_type[idx]
            ],
            agent_type=agent_type.value,
            curr_agent_state=self.curr_agent_state[match_type],
            agent_hist=self.agent_hist[match_type],
            agent_hist_extent=self.agent_hist_extent[match_type],
            agent_hist_len=self.agent_hist_len[match_type],
            agent_fut=self.agent_fut[match_type],
            agent_fut_extent=self.agent_fut_extent[match_type],
            agent_fut_len=self.agent_fut_len[match_type],
            num_neigh=self.num_neigh[match_type],
            neigh_types=self.neigh_types[match_type],
            neigh_hist=self.neigh_hist[match_type],
            neigh_hist_extents=self.neigh_hist_extents[match_type],
            neigh_hist_len=self.neigh_hist_len[match_type],
            neigh_fut=self.neigh_fut[match_type],
            neigh_fut_extents=self.neigh_fut_extents[match_type],
            neigh_fut_len=self.neigh_fut_len[match_type],
            robot_fut=self.robot_fut[match_type]
            if self.robot_fut is not None
            else None,
            robot_fut_len=self.robot_fut_len[match_type]
            if self.robot_fut_len is not None
            else None,
            map_names=[
                name for idx, name in enumerate(self.map_names) if match_type[idx]
            ]
            if self.map_names is not None
            else None,
            maps=self.maps[match_type] if self.maps is not None else None,
            maps_resolution=self.maps_resolution[match_type]
            if self.maps_resolution is not None
            else None,
            vector_maps=[
                vector_map
                for idx, vector_map in enumerate(self.vector_maps)
                if match_type[idx]
            ]
            if self.vector_maps is not None
            else None,
            rasters_from_world_tf=self.rasters_from_world_tf[match_type]
            if self.rasters_from_world_tf is not None
            else None,
            agents_from_world_tf=self.agents_from_world_tf[match_type],
            scene_ids=[
                scene_id
                for idx, scene_id in enumerate(self.scene_ids)
                if match_type[idx]
            ],
            history_pad_dir=self.history_pad_dir,
            extras={key: val[match_type] for key, val in self.extras},
        )


@dataclass
class SceneBatch:
    data_idx: Tensor
    scene_ts: Tensor
    dt: Tensor
    num_agents: Tensor
    agent_type: Tensor
    centered_agent_state: Tensor
    agent_names: List[str]
    agent_hist: Tensor
    agent_hist_extent: Tensor
    agent_hist_len: Tensor
    agent_fut: Tensor
    agent_fut_extent: Tensor
    agent_fut_len: Tensor
    robot_fut: Optional[Tensor]
    robot_fut_len: Optional[Tensor]
    map_names: Optional[Tensor]
    maps: Optional[Tensor]
    maps_resolution: Optional[Tensor]
    vector_maps: Optional[List[VectorMap]]
    rasters_from_world_tf: Optional[Tensor]
    centered_agent_from_world_tf: Tensor
    centered_world_from_agent_tf: Tensor
    scene_ids: Optional[List]
    history_pad_dir: PadDirection
    extras: Dict[str, Tensor]

    def to(self, device) -> None:
        excl_vals = {
            "agent_names",
            "map_names",
            "vector_maps",
            "history_pad_dir",
            "extras",
        }

        for val in vars(self).keys():
            tensor_val = getattr(self, val)
            if val not in excl_vals and tensor_val is not None:
                setattr(self, val, tensor_val.to(device))

        for key, val in self.extras.items():
            self.extras[key] = val.to(device)

    def agent_types(self) -> List[AgentType]:
        unique_types: Tensor = torch.unique(self.agent_type)
        return [AgentType(unique_type.item()) for unique_type in unique_types]

    def for_agent_type(self, agent_type: AgentType) -> SceneBatch:
        match_type = self.agent_type == agent_type
        return SceneBatch(
            data_idx=self.data_idx[match_type],
            scene_ts=self.scene_ts[match_type],
            dt=self.dt[match_type],
            num_agents=self.num_agents[match_type],
            agent_type=self.agent_type[match_type],
            centered_agent_state=self.centered_agent_state[match_type],
            agent_names=[
                agent_name
                for idx, agent_name in enumerate(self.agent_names)
                if match_type[idx]
            ],
            agent_hist=self.agent_hist[match_type],
            agent_hist_extent=self.agent_hist_extent[match_type],
            agent_hist_len=self.agent_hist_len[match_type],
            agent_fut=self.agent_fut[match_type],
            agent_fut_extent=self.agent_fut_extent[match_type],
            agent_fut_len=self.agent_fut_len[match_type],
            robot_fut=self.robot_fut[match_type]
            if self.robot_fut is not None
            else None,
            robot_fut_len=self.robot_fut_len[match_type]
            if self.robot_fut_len is not None
            else None,
            map_names=[
                name for idx, name in enumerate(self.map_names) if match_type[idx]
            ]
            if self.map_names is not None
            else None,
            maps=self.maps[match_type] if self.maps is not None else None,
            maps_resolution=self.maps_resolution[match_type]
            if self.maps_resolution is not None
            else None,
            vector_maps=[
                vector_map
                for idx, vector_map in enumerate(self.vector_maps)
                if match_type[idx]
            ]
            if self.vector_maps is not None
            else None,
            rasters_from_world_tf=self.rasters_from_world_tf[match_type]
            if self.rasters_from_world_tf is not None
            else None,
            centered_agent_from_world_tf=self.centered_agent_from_world_tf[match_type],
            centered_world_from_agent_tf=self.centered_world_from_agent_tf[match_type],
            scene_ids=[
                scene_id
                for idx, scene_id in enumerate(self.scene_ids)
                if match_type[idx]
            ],
            history_pad_dir=self.history_pad_dir,
            extras={key: val[match_type] for key, val in self.extras},
        )
