from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor


class MapMetadata:
    def __init__(
        self,
        name: str,
        shape: Tuple[int, int],
        layers: List[str],
        layer_rgb_groups: Tuple[List[int], List[int], List[int]],
        resolution: float,  # px/m
        map_from_world: np.ndarray,  # Transformation from world coordinates [m] to map coordinates [px]
    ) -> None:
        self.name: str = name
        self.shape: Tuple[int, int] = shape
        self.layers: List[str] = layers
        self.layer_rgb_groups: Tuple[List[int], List[int], List[int]] = layer_rgb_groups
        self.resolution: float = resolution
        self.map_from_world: np.ndarray = map_from_world


class Map:
    def __init__(
        self,
        metadata: MapMetadata,
        data: np.ndarray,
    ) -> None:
        assert data.shape == metadata.shape
        self.metadata: MapMetadata = metadata
        self.data: np.ndarray = data

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @staticmethod
    def to_img(
        map_arr: Tensor,
        idx_groups: Optional[Tuple[List[int], List[int], List[int]]] = None,
    ) -> Tensor:
        if idx_groups is None:
            return map_arr.permute(1, 2, 0).numpy()

        return torch.stack(
            [
                torch.amax(map_arr[idx_groups[0]], dim=0),
                torch.amax(map_arr[idx_groups[1]], dim=0),
                torch.amax(map_arr[idx_groups[2]], dim=0),
            ],
            dim=-1,
        ).numpy()
