import pickle
from math import ceil, floor
from pathlib import Path
from typing import Callable, Dict, Final, List, Optional, Tuple

import dill
import kornia
import numpy as np
import pandas as pd
import torch
import zarr

from trajdata.augmentation.augmentation import Augmentation, DatasetAugmentation
from trajdata.caching.scene_cache import SceneCache
from trajdata.data_structures.agent import AgentMetadata, FixedExtent
from trajdata.data_structures.map import Map, MapMetadata
from trajdata.data_structures.scene_metadata import Scene
from trajdata.utils import arr_utils

STATE_COLS: Final[List[str]] = ["x", "y", "vx", "vy", "ax", "ay"]
EXTENT_COLS: Final[List[str]] = ["length", "width", "height"]


class DataFrameCache(SceneCache):
    def __init__(
        self,
        cache_path: Path,
        scene: Scene,
        scene_ts: int,
        augmentations: Optional[List[Augmentation]] = None,
    ) -> None:
        """
        Data cache primarily based on pandas DataFrames,
        with Feather for fast agent data serialization
        and pickle for miscellaneous supporting objects.
        Maps are pre-rasterized and stored as Zarr arrays.
        """
        super().__init__(cache_path, scene, scene_ts, augmentations)

        self.agent_data_path: Path = self.scene_dir / "agent_data.feather"

        self._load_agent_data()
        self._get_and_reorder_col_idxs()

        if augmentations:
            dataset_augments: List[DatasetAugmentation] = [
                augment
                for augment in augmentations
                if isinstance(augment, DatasetAugmentation)
            ]
            for aug in dataset_augments:
                aug.apply(self.scene_data_df)

    # AGENT STATE DATA
    def _get_and_reorder_col_idxs(self) -> None:
        avail_extent_cols: List[str] = [
            col for col in self.scene_data_df.columns if col in EXTENT_COLS
        ]
        avail_heading_cols: List[str] = (
            ["heading"]
            if "heading" in self.scene_data_df.columns
            else ["sin_heading", "cos_heading"]
        )
        self.scene_data_df = self.scene_data_df[
            STATE_COLS + avail_heading_cols + avail_extent_cols
        ]

        self.column_dict: Dict[str, int] = {
            val: idx for idx, val in enumerate(self.scene_data_df.columns)
        }

        self.pos_cols = [self.column_dict["x"], self.column_dict["y"]]
        self.vel_cols = [self.column_dict["vx"], self.column_dict["vy"]]
        self.acc_cols = [self.column_dict["ax"], self.column_dict["ay"]]
        if "sin_heading" in self.column_dict:
            self.heading_cols = [
                self.column_dict["sin_heading"],
                self.column_dict["cos_heading"],
            ]
        else:
            self.heading_cols = [self.column_dict["heading"]]

        self.state_cols = (
            self.pos_cols + self.vel_cols + self.acc_cols + self.heading_cols
        )
        self.state_dim = len(self.state_cols)

        self.extent_cols: List[int] = list()
        for extent_name in ["length", "width", "height"]:
            if extent_name in self.column_dict:
                self.extent_cols.append(self.column_dict[extent_name])

    def _load_agent_data(self) -> pd.DataFrame:
        self.scene_data_df: pd.DataFrame = pd.read_feather(
            self.agent_data_path, use_threads=False
        ).set_index(["agent_id", "scene_ts"])

        with open(self.scene_dir / "scene_index.pkl", "rb") as f:
            self.index_dict: Dict[Tuple[str, int], int] = pickle.load(f)

    @staticmethod
    def save_agent_data(
        agent_data: pd.DataFrame,
        cache_path: Path,
        scene: Scene,
    ) -> None:
        scene_cache_dir: Path = cache_path / scene.env_name / scene.name
        scene_cache_dir.mkdir(parents=True, exist_ok=True)

        index_dict: Dict[Tuple[str, int], int] = {
            val: idx for idx, val in enumerate(agent_data.index)
        }
        with open(scene_cache_dir / "scene_index.pkl", "wb") as f:
            pickle.dump(index_dict, f)

        agent_data.reset_index().to_feather(scene_cache_dir / "agent_data.feather")

    def get_value(self, agent_id: str, scene_ts: int, attribute: str) -> float:
        return self.scene_data_df.iat[
            self.index_dict[(agent_id, scene_ts)], self.column_dict[attribute]
        ]

    def get_state(self, agent_id: str, scene_ts: int) -> np.ndarray:
        # Setting copy=True so that the returned state is untouched by any
        # future transformations (e.g., by a call to the transform_data function).
        return self.scene_data_df.iloc[
            self.index_dict[(agent_id, scene_ts)], : self.state_dim
        ].to_numpy(copy=True)

    def transform_data(self, **kwargs) -> None:
        if "shift_mean_to" in kwargs:
            # This standardizes the scene to be relative to the agent being predicted.
            self.scene_data_df.iloc[:, : self.state_dim] -= kwargs["shift_mean_to"]

        if "rotate_by" in kwargs:
            # This rotates the scene so that the predicted agent's current
            # heading aligns with the x-axis.
            agent_heading: float = kwargs["rotate_by"]
            self.rot_matrix: np.ndarray = np.array(
                [
                    [np.cos(agent_heading), -np.sin(agent_heading)],
                    [np.sin(agent_heading), np.cos(agent_heading)],
                ]
            )
            self.scene_data_df.iloc[:, self.pos_cols] = (
                self.scene_data_df.iloc[:, self.pos_cols].to_numpy() @ self.rot_matrix
            )
            self.scene_data_df.iloc[:, self.vel_cols] = (
                self.scene_data_df.iloc[:, self.vel_cols].to_numpy() @ self.rot_matrix
            )
            self.scene_data_df.iloc[:, self.acc_cols] = (
                self.scene_data_df.iloc[:, self.acc_cols].to_numpy() @ self.rot_matrix
            )

        if "sincos_heading" in kwargs:
            self.scene_data_df["sin_heading"] = np.sin(self.scene_data_df["heading"])
            self.scene_data_df["cos_heading"] = np.cos(self.scene_data_df["heading"])
            self.scene_data_df.drop(columns=["heading"], inplace=True)
            self._get_and_reorder_col_idxs()

    def interpolate_data(self, desired_dt: float, method: str = "linear") -> None:
        dt_ratio: float = self.scene.env_metadata.dt / desired_dt
        if not dt_ratio.is_integer():
            raise ValueError(
                f"{str(self.scene)}'s dt of {self.scene.dt}s "
                f"is not divisible by the desired dt {desired_dt}s."
            )

        dt_factor: int = int(dt_ratio)

        agent_info_dict: Dict[str, AgentMetadata] = {
            agent.name: agent for agent in self.scene.agents
        }
        new_index: pd.MultiIndex = pd.MultiIndex.from_tuples(
            [
                (agent_name, scene_ts)
                for agent_name in self.scene_data_df.index.unique(level=0)
                for scene_ts in range(
                    # These have already been multiplied by dt_factor earlier.
                    agent_info_dict[agent_name].first_timestep,
                    agent_info_dict[agent_name].last_timestep + 1,
                )
            ],
            names=["agent_id", "scene_ts"],
        )

        interpolated_df: pd.DataFrame = pd.DataFrame(
            index=new_index, columns=self.scene_data_df.columns
        )
        interpolated_df = interpolated_df.astype(self.scene_data_df.dtypes.to_dict())

        scene_data: np.ndarray = self.scene_data_df.to_numpy()
        unwrapped_heading: np.ndarray = np.unwrap(self.scene_data_df["heading"])

        # Getting the data initially in the new df, making sure to unwrap angles above
        # in preparation for interpolation.
        scene_data_idxs: np.ndarray = np.nonzero(
            new_index.get_level_values("scene_ts") % dt_factor == 0
        )[0]
        interpolated_df.iloc[scene_data_idxs] = scene_data
        interpolated_df.iloc[
            scene_data_idxs, self.column_dict["heading"]
        ] = unwrapped_heading

        # Interpolation.
        interpolated_df.interpolate(method=method, limit_area="inside", inplace=True)

        # Wrapping angles back to [-pi, pi).
        interpolated_df.iloc[:, self.column_dict["heading"]] = arr_utils.angle_wrap(
            interpolated_df.iloc[:, self.column_dict["heading"]]
        )

        self.scene_data_df = interpolated_df
        self.index_dict: Dict[Tuple[str, int], int] = {
            val: idx for idx, val in enumerate(self.scene_data_df.index)
        }

    def get_agent_history(
        self,
        agent_info: AgentMetadata,
        scene_ts: int,
        history_sec: Tuple[Optional[float], Optional[float]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        # We don't have to check the mins here because our data_index filtering in dataset.py already
        # took care of it.
        first_index_incl: int
        last_index_incl: int = self.index_dict[(agent_info.name, scene_ts)]
        if history_sec[1] is not None:
            max_history: int = floor(history_sec[1] / self.dt)
            first_index_incl = self.index_dict[
                (
                    agent_info.name,
                    max(scene_ts - max_history, agent_info.first_timestep),
                )
            ]
        else:
            first_index_incl = self.index_dict[
                (agent_info.name, agent_info.first_timestep)
            ]

        agent_history_df: pd.DataFrame = self.scene_data_df.iloc[
            first_index_incl : last_index_incl + 1
        ]

        agent_extent_np: np.ndarray
        if isinstance(agent_info.extent, FixedExtent):
            agent_extent_np = agent_info.extent.get_extents(
                self.scene_ts - agent_history_df.shape[0] + 1, self.scene_ts
            )
        else:
            agent_extent_np = agent_history_df.iloc[:, self.extent_cols].to_numpy()

        return agent_history_df.iloc[:, : self.state_dim].to_numpy(), agent_extent_np

    def get_agent_future(
        self,
        agent_info: AgentMetadata,
        scene_ts: int,
        future_sec: Tuple[Optional[float], Optional[float]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        # We don't have to check the mins here because our data_index filtering in dataset.py already
        # took care of it.
        if scene_ts >= agent_info.last_timestep:
            # Extent shape = 3
            return np.zeros((0, self.state_dim)), np.zeros((0, 3))

        first_index_incl: int = self.index_dict[(agent_info.name, scene_ts + 1)]
        last_index_incl: int
        if future_sec[1] is not None:
            max_future = floor(future_sec[1] / self.dt)
            last_index_incl = self.index_dict[
                (agent_info.name, min(scene_ts + max_future, agent_info.last_timestep))
            ]
        else:
            last_index_incl = self.index_dict[
                (agent_info.name, agent_info.last_timestep)
            ]

        agent_future_df: pd.DataFrame = self.scene_data_df.iloc[
            first_index_incl : last_index_incl + 1
        ]

        agent_extent_np: np.ndarray
        if isinstance(agent_info.extent, FixedExtent):
            agent_extent_np: np.ndarray = agent_info.extent.get_extents(
                self.scene_ts + 1, self.scene_ts + agent_future_df.shape[0]
            )
        else:
            agent_extent_np = agent_future_df.iloc[:, self.extent_cols].to_numpy()

        return agent_future_df.iloc[:, : self.state_dim].to_numpy(), agent_extent_np

    def get_positions_at(
        self, scene_ts: int, agents: List[AgentMetadata]
    ) -> np.ndarray:
        rows = [self.index_dict[(agent.name, scene_ts)] for agent in agents]
        return self.scene_data_df.iloc[rows, self.pos_cols].to_numpy()

    def get_agents_history(
        self,
        scene_ts: int,
        agents: List[AgentMetadata],
        history_sec: Tuple[Optional[float], Optional[float]],
    ) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
        first_timesteps = np.array(
            [agent.first_timestep for agent in agents], dtype=np.long
        )
        if history_sec[1] is not None:
            max_history: int = floor(history_sec[1] / self.dt)
            first_timesteps = np.maximum(scene_ts - max_history, first_timesteps)

        first_index_incl: np.ndarray = np.array(
            [
                self.index_dict[(agent.name, first_timesteps[idx])]
                for idx, agent in enumerate(agents)
            ],
            dtype=np.long,
        )
        last_index_incl: np.ndarray = np.array(
            [self.index_dict[(agent.name, scene_ts)] for agent in agents], dtype=np.long
        )

        concat_idxs = arr_utils.vrange(first_index_incl, last_index_incl + 1)
        neighbor_data_df: pd.DataFrame = self.scene_data_df.iloc[concat_idxs, :]

        neighbor_history_lens_np = last_index_incl - first_index_incl + 1

        neighbor_histories_np = neighbor_data_df.iloc[:, : self.state_dim].to_numpy()
        # The last one will always be empty because of what cumsum returns.
        neighbor_histories: List[np.ndarray] = np.vsplit(
            neighbor_histories_np, neighbor_history_lens_np.cumsum()
        )[:-1]

        neighbor_extents: List[np.ndarray]
        if self.extent_cols:
            neighbor_extents_np = neighbor_data_df.iloc[:, self.extent_cols].to_numpy()
            # The last one will always be empty because of what cumsum returns.
            neighbor_extents = np.vsplit(
                neighbor_extents_np, neighbor_history_lens_np.cumsum()
            )[:-1]
        else:
            neighbor_extents = [
                agent.extent.get_extents(
                    self.scene_ts - neighbor_history_lens_np[idx].item() + 1,
                    self.scene_ts,
                )
                for idx, agent in enumerate(agents)
            ]

        return (
            neighbor_histories,
            neighbor_extents,
            neighbor_history_lens_np,
        )

    def get_agents_future(
        self,
        scene_ts: int,
        agents: List[AgentMetadata],
        future_sec: Tuple[Optional[float], Optional[float]],
    ) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
        last_timesteps = np.array(
            [agent.last_timestep for agent in agents], dtype=np.long
        )

        first_timesteps = np.array(
            [agent.first_timestep for agent in agents], dtype=np.long
        )
        first_timesteps = np.minimum(scene_ts + 1, last_timesteps)

        if future_sec[1] is not None:
            max_future: int = floor(future_sec[1] / self.dt)
            last_timesteps = np.minimum(scene_ts + max_future, last_timesteps)

        first_index_incl: np.ndarray = np.array(
            [
                self.index_dict[(agent.name, first_timesteps[idx])]
                for idx, agent in enumerate(agents)
            ],
            dtype=np.long,
        )
        last_index_incl: np.ndarray = np.array(
            [
                self.index_dict[(agent.name, last_timesteps[idx])]
                for idx, agent in enumerate(agents)
            ],
            dtype=np.long,
        )

        concat_idxs = arr_utils.vrange(first_index_incl, last_index_incl + 1)
        neighbor_data_df: pd.DataFrame = self.scene_data_df.iloc[concat_idxs, :]

        neighbor_future_lens_np = last_index_incl - first_index_incl + 1

        neighbor_futures_np = neighbor_data_df.iloc[:, : self.state_dim].to_numpy()
        # The last one will always be empty because of what cumsum returns.
        neighbor_futures: List[np.ndarray] = np.vsplit(
            neighbor_futures_np, neighbor_future_lens_np.cumsum()
        )[:-1]

        neighbor_extents: List[np.ndarray]
        if self.extent_cols:
            neighbor_extents_np = neighbor_data_df.iloc[:, self.extent_cols].to_numpy()
            # The last one will always be empty because of what cumsum returns.
            neighbor_extents = np.vsplit(
                neighbor_extents_np, neighbor_future_lens_np.cumsum()
            )[:-1]
        else:
            neighbor_extents = [
                agent.extent.get_extents(
                    self.scene_ts - neighbor_future_lens_np[idx].item() + 1,
                    self.scene_ts,
                )
                for idx, agent in enumerate(agents)
            ]

        return (
            neighbor_futures,
            neighbor_extents,
            neighbor_future_lens_np,
        )

    # MAPS
    @staticmethod
    def get_maps_path(cache_path: Path, env_name: str) -> Path:
        return cache_path / env_name / "maps"

    @staticmethod
    def are_maps_cached(cache_path: Path, env_name: str) -> bool:
        return DataFrameCache.get_maps_path(cache_path, env_name).is_dir()

    @staticmethod
    def is_map_cached(cache_path: Path, env_name: str, map_name: str) -> bool:
        maps_path: Path = DataFrameCache.get_maps_path(cache_path, env_name)
        metadata_file: Path = maps_path / f"{map_name}_metadata.dill"
        map_file: Path = maps_path / f"{map_name}.zarr"
        return maps_path.is_dir() and metadata_file.is_file() and map_file.is_file()

    @staticmethod
    def cache_map(cache_path: Path, map_obj: Map, env_name: str) -> None:
        maps_path: Path = DataFrameCache.get_maps_path(cache_path, env_name)
        maps_path.mkdir(parents=True, exist_ok=True)

        map_file: Path = maps_path / f"{map_obj.metadata.name}.zarr"
        zarr.save(map_file, map_obj.data)

        metadata_file: Path = maps_path / f"{map_obj.metadata.name}_metadata.dill"
        with open(metadata_file, "wb") as f:
            dill.dump(map_obj.metadata, f)

    @staticmethod
    def cache_map_layers(
        cache_path: Path,
        map_info: MapMetadata,
        layer_fn: Callable[[str], np.ndarray],
        env_name: str,
    ) -> None:
        maps_path: Path = DataFrameCache.get_maps_path(cache_path, env_name)
        maps_path.mkdir(parents=True, exist_ok=True)

        map_file: Path = maps_path / f"{map_info.name}.zarr"
        disk_data = zarr.open_array(map_file, mode="w", shape=map_info.shape)
        for idx, layer_name in enumerate(map_info.layers):
            disk_data[idx] = layer_fn(layer_name)

        metadata_file: Path = maps_path / f"{map_info.name}_metadata.dill"
        with open(metadata_file, "wb") as f:
            dill.dump(map_info, f)

    def pad_map_patch(
        self,
        patch: np.ndarray,
        # top, bot, left, right
        patch_sides: Tuple[int, int, int, int],
        patch_size: int,
        map_dims: Tuple[int, int, int],
    ) -> np.ndarray:
        if patch.shape[-2:] == (patch_size, patch_size):
            return patch

        top, bot, left, right = patch_sides
        channels, height, width = map_dims

        # If we're off the map, just return zeros in the
        # desired size of the patch.
        if bot <= 0 or top >= height or right <= 0 or left >= width:
            return np.zeros((channels, patch_size, patch_size))

        pad_top, pad_bot, pad_left, pad_right = 0, 0, 0, 0
        if top < 0:
            pad_top = 0 - top
        if bot >= height:
            pad_bot = bot - height
        if left < 0:
            pad_left = 0 - left
        if right >= width:
            pad_right = right - width

        return np.pad(patch, [(0, 0), (pad_top, pad_bot), (pad_left, pad_right)])

    def load_map_patch(
        self,
        world_x: float,
        world_y: float,
        desired_patch_size: int,
        resolution: int,
        offset_xy: Tuple[float, float],
        agent_heading: float,
        return_rgb: bool,
        rot_pad_factor: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        maps_path: Path = DataFrameCache.get_maps_path(self.path, self.scene.env_name)

        metadata_file: Path = maps_path / f"{self.scene.location}_metadata.dill"
        with open(metadata_file, "rb") as f:
            map_info: MapMetadata = dill.load(f)

        raster_from_world_tf: np.ndarray = map_info.map_from_world
        map_coords: np.ndarray = map_info.map_from_world @ np.array(
            [world_x, world_y, 1.0]
        )
        map_x, map_y = map_coords[0].item(), map_coords[1].item()

        raster_from_world_tf = (
            np.array(
                [
                    [1.0, 0.0, -map_x],
                    [0.0, 1.0, -map_y],
                    [0.0, 0.0, 1.0],
                ]
            )
            @ raster_from_world_tf
        )

        data_patch_size: int = ceil(
            desired_patch_size * map_info.resolution / resolution
        )

        # Incorporating offsets.
        if offset_xy != (0.0, 0.0):
            # x is negative here because I am moving the map
            # center so that the agent ends up where the user wishes
            # (the agent is pinned from the end user's perspective).
            map_offset: Tuple[float, float] = (
                -offset_xy[0] * data_patch_size // 2,
                offset_xy[1] * data_patch_size // 2,
            )

            rotated_offset: np.ndarray = (
                arr_utils.rotation_matrix(agent_heading) @ map_offset
            )

            off_x = rotated_offset[0]
            off_y = rotated_offset[1]

            map_x += off_x
            map_y += off_y

            raster_from_world_tf = (
                np.array(
                    [
                        [1.0, 0.0, -off_x],
                        [0.0, 1.0, -off_y],
                        [0.0, 0.0, 1.0],
                    ]
                )
                @ raster_from_world_tf
            )

        # Ensuring the size is divisible by two so that the // 2 below does not
        # chop any information off.
        data_with_rot_pad_size: int = ceil((rot_pad_factor * data_patch_size) / 2) * 2

        map_file: Path = maps_path / f"{map_info.name}.zarr"
        disk_data = zarr.open_array(map_file, mode="r")

        map_x = round(map_x)
        map_y = round(map_y)

        half_extent: int = data_with_rot_pad_size // 2

        top: int = map_y - half_extent
        bot: int = map_y + half_extent
        left: int = map_x - half_extent
        right: int = map_x + half_extent

        data_patch: np.ndarray = self.pad_map_patch(
            disk_data[
                ...,
                max(top, 0) : min(bot, disk_data.shape[1]),
                max(left, 0) : min(right, disk_data.shape[2]),
            ],
            (top, bot, left, right),
            data_with_rot_pad_size,
            disk_data.shape,
        )

        if return_rgb:
            rgb_groups = map_info.layer_rgb_groups
            data_patch = np.stack(
                [
                    np.amax(data_patch[rgb_groups[0]], axis=0),
                    np.amax(data_patch[rgb_groups[1]], axis=0),
                    np.amax(data_patch[rgb_groups[2]], axis=0),
                ],
            )

        if desired_patch_size != data_patch_size:
            scale_factor: float = desired_patch_size / data_patch_size
            data_patch = (
                kornia.geometry.rescale(
                    torch.from_numpy(data_patch).unsqueeze(0),
                    scale_factor,
                    # Default align_corners value, just putting it to remove warnings
                    align_corners=False,
                    antialias=True,
                )
                .squeeze(0)
                .numpy()
            )

            raster_from_world_tf = (
                np.array(
                    [
                        [1 / scale_factor, 0.0, 0.0],
                        [0.0, 1 / scale_factor, 0.0],
                        [0.0, 0.0, 1.0],
                    ]
                )
                @ raster_from_world_tf
            )

        return data_patch, raster_from_world_tf
