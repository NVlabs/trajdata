from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch import Tensor

from trajdata.data_structures.agent import AgentType
from trajdata.data_structures.state import StateTensor
from trajdata.maps import VectorMap
from trajdata.utils.arr_utils import PadDirection


@dataclass
class AgentBatch:
    data_idx: Tensor
    scene_ts: Tensor
    dt: Tensor
    agent_name: List[str]
    agent_type: Tensor
    curr_agent_state: StateTensor
    agent_hist: StateTensor
    agent_hist_extent: Tensor
    agent_hist_len: Tensor
    agent_fut: StateTensor
    agent_fut_extent: Tensor
    agent_fut_len: Tensor
    num_neigh: Tensor
    neigh_types: Tensor
    neigh_hist: StateTensor
    neigh_hist_extents: Tensor
    neigh_hist_len: Tensor
    neigh_fut: StateTensor
    neigh_fut_extents: Tensor
    neigh_fut_len: Tensor
    robot_fut: Optional[StateTensor]
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
        return self.filter_batch(match_type)

    def filter_batch(self, filter_mask: torch.Tensor) -> AgentBatch:
        """Build a new batch with elements for which filter_mask[i] == True."""

        # Some of the tensors might be on different devices, so we define some convenience functions
        # to make sure the filter_mask is always on the same device as the tensor we are indexing.
        filter_mask_dict = {}
        filter_mask_dict["cpu"] = filter_mask.to("cpu")
        filter_mask_dict[str(self.agent_hist.device)] = filter_mask.to(
            self.agent_hist.device
        )

        _filter = lambda tensor: tensor[filter_mask_dict[str(tensor.device)]]
        _filter_tensor_or_list = lambda tensor_or_list: (
            _filter(tensor_or_list)
            if isinstance(tensor_or_list, torch.Tensor)
            else type(tensor_or_list)(
                [
                    el
                    for idx, el in enumerate(tensor_or_list)
                    if filter_mask_dict["cpu"][idx]
                ]
            )
        )

        return AgentBatch(
            data_idx=_filter(self.data_idx),
            scene_ts=_filter(self.scene_ts),
            dt=_filter(self.dt),
            agent_name=_filter_tensor_or_list(self.agent_name),
            agent_type=_filter(self.agent_type),
            curr_agent_state=_filter(self.curr_agent_state),
            agent_hist=_filter(self.agent_hist),
            agent_hist_extent=_filter(self.agent_hist_extent),
            agent_hist_len=_filter(self.agent_hist_len),
            agent_fut=_filter(self.agent_fut),
            agent_fut_extent=_filter(self.agent_fut_extent),
            agent_fut_len=_filter(self.agent_fut_len),
            num_neigh=_filter(self.num_neigh),
            neigh_types=_filter(self.neigh_types),
            neigh_hist=_filter(self.neigh_hist),
            neigh_hist_extents=_filter(self.neigh_hist_extents),
            neigh_hist_len=_filter(self.neigh_hist_len),
            neigh_fut=_filter(self.neigh_fut),
            neigh_fut_extents=_filter(self.neigh_fut_extents),
            neigh_fut_len=_filter(self.neigh_fut_len),
            robot_fut=_filter(self.robot_fut) if self.robot_fut is not None else None,
            robot_fut_len=_filter(self.robot_fut_len)
            if self.robot_fut_len is not None
            else None,
            map_names=_filter_tensor_or_list(self.map_names)
            if self.map_names is not None
            else None,
            maps=_filter(self.maps) if self.maps is not None else None,
            maps_resolution=_filter(self.maps_resolution)
            if self.maps_resolution is not None
            else None,
            vector_maps=_filter(self.vector_maps)
            if self.vector_maps is not None
            else None,
            rasters_from_world_tf=_filter(self.rasters_from_world_tf)
            if self.rasters_from_world_tf is not None
            else None,
            agents_from_world_tf=_filter(self.agents_from_world_tf),
            scene_ids=_filter_tensor_or_list(self.scene_ids),
            history_pad_dir=self.history_pad_dir,
            extras={
                key: _filter_tensor_or_list(val) for key, val in self.extras.items()
            },
        )


@dataclass
class SceneBatch:
    data_idx: Tensor
    scene_ts: Tensor
    dt: Tensor
    num_agents: Tensor
    agent_type: Tensor
    centered_agent_state: StateTensor
    agent_names: List[str]
    agent_hist: StateTensor
    agent_hist_extent: Tensor
    agent_hist_len: Tensor
    agent_fut: StateTensor
    agent_fut_extent: Tensor
    agent_fut_len: Tensor
    robot_fut: Optional[StateTensor]
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
            "scene_ids",
            "extras",
        }

        for val in vars(self).keys():
            tensor_val = getattr(self, val)
            if val not in excl_vals and tensor_val is not None:
                setattr(self, val, tensor_val.to(device))

        for key, val in self.extras.items():
            # Allow for custom .to() method for objects that define a __to__ function.
            if hasattr(val, "__to__"):
                self.extras[key] = val.__to__(device, non_blocking=True)
            else:
                self.extras[key] = val.to(device, non_blocking=True)

    def agent_types(self) -> List[AgentType]:
        unique_types: Tensor = torch.unique(self.agent_type)
        return [
            AgentType(unique_type.item())
            for unique_type in unique_types
            if unique_type >= 0
        ]

    def for_agent_type(self, agent_type: AgentType) -> SceneBatch:
        match_type = self.agent_type == agent_type
        return self.filter_batch(match_type)

    def filter_batch(self, filter_mask: torch.tensor) -> SceneBatch:
        """Build a new batch with elements for which filter_mask[i] == True."""

        # Some of the tensors might be on different devices, so we define some convenience functions
        # to make sure the filter_mask is always on the same device as the tensor we are indexing.
        filter_mask_dict = {}
        filter_mask_dict["cpu"] = filter_mask.to("cpu")
        filter_mask_dict[str(self.agent_hist.device)] = filter_mask.to(
            self.agent_hist.device
        )

        _filter = lambda tensor: tensor[filter_mask_dict[str(tensor.device)]]
        _filter_tensor_or_list = lambda tensor_or_list: (
            _filter(tensor_or_list)
            if isinstance(tensor_or_list, torch.Tensor)
            else type(tensor_or_list)(
                [
                    el
                    for idx, el in enumerate(tensor_or_list)
                    if filter_mask_dict["cpu"][idx]
                ]
            )
        )

        return SceneBatch(
            data_idx=_filter(self.data_idx),
            scene_ts=_filter(self.scene_ts),
            dt=_filter(self.dt),
            num_agents=_filter(self.num_agents),
            agent_type=_filter(self.agent_type),
            centered_agent_state=_filter(self.centered_agent_state),
            agent_hist=_filter(self.agent_hist),
            agent_hist_extent=_filter(self.agent_hist_extent),
            agent_hist_len=_filter(self.agent_hist_len),
            agent_fut=_filter(self.agent_fut),
            agent_fut_extent=_filter(self.agent_fut_extent),
            agent_fut_len=_filter(self.agent_fut_len),
            robot_fut=_filter(self.robot_fut) if self.robot_fut is not None else None,
            robot_fut_len=_filter(self.robot_fut_len)
            if self.robot_fut_len is not None
            else None,
            map_names=_filter_tensor_or_list(self.map_names)
            if self.map_names is not None
            else None,
            maps=_filter(self.maps) if self.maps is not None else None,
            maps_resolution=_filter(self.maps_resolution)
            if self.maps_resolution is not None
            else None,
            vector_maps=_filter(self.vector_maps)
            if self.vector_maps is not None
            else None,
            rasters_from_world_tf=_filter(self.rasters_from_world_tf)
            if self.rasters_from_world_tf is not None
            else None,
            centered_agent_from_world_tf=_filter(self.centered_agent_from_world_tf),
            centered_world_from_agent_tf=_filter(self.centered_world_from_agent_tf),
            scene_ids=_filter_tensor_or_list(self.scene_ids),
            history_pad_dir=self.history_pad_dir,
            extras={
                key: _filter_tensor_or_list(val, filter_mask)
                for key, val in self.extras.items()
            },
        )

    def to_agent_batch(self, agent_inds: torch.Tensor) -> AgentBatch:
        """
        Converts SeceneBatch to AgentBatch for agents defined by `agent_inds`.

        self.extras will be simply copied over, any custom conversion must be
        implemented externally.
        """

        batch_size = self.agent_hist.shape[0]
        num_agents = self.agent_hist.shape[1]

        if agent_inds.ndim != 1 or agent_inds.shape[0] != batch_size:
            raise ValueError("Wrong shape for agent_inds, expected [batch_size].")

        if (agent_inds < 0).any() or (agent_inds >= num_agents).any():
            raise ValueError("Invalid agent index")

        batch_inds = torch.arange(batch_size)
        others_mask = torch.ones((batch_size, num_agents), dtype=torch.bool)
        others_mask[batch_inds, agent_inds] = False
        index_agent = lambda x: x[batch_inds, agent_inds] if x is not None else None
        index_agent_list = (
            lambda xlist: [x[ind] for x, ind in zip(xlist, agent_inds)]
            if xlist is not None
            else None
        )
        index_neighbors = lambda x: x[others_mask].reshape(
            [
                batch_size,
                num_agents - 1,
            ]
            + list(x.shape[2:])
        )

        return AgentBatch(
            data_idx=self.data_idx,
            scene_ts=self.scene_ts,
            dt=self.dt,
            agent_name=index_agent_list(self.agent_names),
            agent_type=index_agent(self.agent_type),
            curr_agent_state=self.centered_agent_state,  # TODO this is not actually the agent but the `global` coordinate frame
            agent_hist=index_agent(self.agent_hist),
            agent_hist_extent=index_agent(self.agent_hist_extent),
            agent_hist_len=index_agent(self.agent_hist_len),
            agent_fut=index_agent(self.agent_fut),
            agent_fut_extent=index_agent(self.agent_fut_extent),
            agent_fut_len=index_agent(self.agent_fut_len),
            num_neigh=self.num_agents - 1,
            neigh_types=index_neighbors(self.agent_type),
            neigh_hist=index_neighbors(self.agent_hist),
            neigh_hist_extents=index_neighbors(self.agent_hist_extent),
            neigh_hist_len=index_neighbors(self.agent_hist_len),
            neigh_fut=index_neighbors(self.agent_fut),
            neigh_fut_extents=index_neighbors(self.agent_fut_extent),
            neigh_fut_len=index_neighbors(self.agent_fut_len),
            robot_fut=self.robot_fut,
            robot_fut_len=self.robot_fut_len,
            map_names=index_agent_list(self.map_names),
            maps=index_agent(self.maps),
            vector_maps=index_agent(self.vector_maps),
            maps_resolution=index_agent(self.maps_resolution),
            rasters_from_world_tf=index_agent(self.rasters_from_world_tf),
            agents_from_world_tf=self.centered_agent_from_world_tf,
            scene_ids=self.scene_ids,
            history_pad_dir=self.history_pad_dir,
            extras=self.extras,
        )
