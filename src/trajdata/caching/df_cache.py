from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trajdata.maps import (
        RasterizedMap,
        RasterizedMapMetadata,
        VectorMap,
    )
    from trajdata.maps.map_kdtree import MapElementKDTree

import pickle
from math import ceil, floor
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Tuple

import dill
import kornia
import numpy as np
import pandas as pd
import torch
import zarr

from trajdata.augmentation.augmentation import Augmentation, DatasetAugmentation
from trajdata.caching.scene_cache import SceneCache
from trajdata.data_structures.agent import AgentMetadata, FixedExtent
from trajdata.data_structures.scene_metadata import Scene
from trajdata.maps.traffic_light_status import TrafficLightStatus
from trajdata.utils import arr_utils, df_utils, raster_utils

STATE_COLS: Final[List[str]] = ["x", "y", "vx", "vy", "ax", "ay"]
EXTENT_COLS: Final[List[str]] = ["length", "width", "height"]


class DataFrameCache(SceneCache):
    def __init__(
        self,
        cache_path: Path,
        scene: Scene,
        augmentations: Optional[List[Augmentation]] = None,
    ) -> None:
        """
        Data cache primarily based on pandas DataFrames,
        with Feather for fast agent data serialization
        and pickle for miscellaneous supporting objects.
        Maps are pre-rasterized and stored as Zarr arrays.
        """
        super().__init__(cache_path, scene, augmentations)

        agent_data_path: Path = self.scene_dir / DataFrameCache._agent_data_file(
            scene.dt
        )
        if not agent_data_path.exists():
            # Load the original dt agent data and then
            # interpolate it to the desired dt.
            self._load_agent_data(scene.env_metadata.dt)
            self.interpolate_data(scene.dt)
        else:
            # Load the data with the desired dt.
            self._load_agent_data(scene.dt)

        # Setting default data transformation parameters.
        self.reset_transforms()

        self._kdtrees = None

        if augmentations:
            dataset_augments: List[DatasetAugmentation] = [
                augment
                for augment in augmentations
                if isinstance(augment, DatasetAugmentation)
            ]
            for aug in dataset_augments:
                aug.apply(self.scene_data_df)

    @staticmethod
    def _agent_data_file(scene_dt: float) -> str:
        return f"agent_data_dt{scene_dt:.2f}.feather"

    @staticmethod
    def _agent_data_index_file(scene_dt: float) -> str:
        return f"scene_index_dt{scene_dt:.2f}.pkl"

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
        self._state_dim = len(self.state_cols)

        # While it may seem that obs_dim == _state_dim, obs_dim is meant to
        # track the output dimension of states (which could differ via
        # returing sin and cos of heading rather than just heading itself).
        self.obs_dim = self._state_dim

        self.extent_cols: List[int] = list()
        for extent_name in ["length", "width", "height"]:
            if extent_name in self.column_dict:
                self.extent_cols.append(self.column_dict[extent_name])

    def _load_agent_data(self, scene_dt: float) -> None:
        self.scene_data_df: pd.DataFrame = pd.read_feather(
            self.scene_dir / DataFrameCache._agent_data_file(scene_dt),
            use_threads=False,
        ).set_index(["agent_id", "scene_ts"])

        with open(
            self.scene_dir / DataFrameCache._agent_data_index_file(scene_dt), "rb"
        ) as f:
            self.index_dict: Dict[Tuple[str, int], int] = pickle.load(f)

        self._get_and_reorder_col_idxs()

    def write_cache_to_disk(self) -> None:
        with open(
            self.scene_dir / DataFrameCache._agent_data_index_file(self.dt), "wb"
        ) as f:
            pickle.dump(self.index_dict, f)

        self.scene_data_df.reset_index().to_feather(
            self.scene_dir / DataFrameCache._agent_data_file(self.dt)
        )

    @staticmethod
    def save_agent_data(
        agent_data: pd.DataFrame,
        cache_path: Path,
        scene: Scene,
    ) -> None:
        scene_cache_dir: Path = DataFrameCache.scene_cache_dir(
            cache_path, scene.env_name, scene.name
        )
        scene_cache_dir.mkdir(parents=True, exist_ok=True)

        index_dict: Dict[Tuple[str, int], int] = {
            val: idx for idx, val in enumerate(agent_data.index)
        }
        with open(
            scene_cache_dir / DataFrameCache._agent_data_index_file(scene.dt), "wb"
        ) as f:
            pickle.dump(index_dict, f)

        agent_data.reset_index().to_feather(
            scene_cache_dir / DataFrameCache._agent_data_file(scene.dt)
        )

    def get_value(self, agent_id: str, scene_ts: int, attribute: str) -> float:
        col_idx: int = self.column_dict[attribute]
        value: float = self.scene_data_df.iat[
            self.index_dict[(agent_id, scene_ts)], col_idx
        ]

        if "y" in attribute:
            x_value: float = self.scene_data_df.iat[
                self.index_dict[(agent_id, scene_ts)], col_idx - 1
            ]

            transformed_pair: np.ndarray = self._transform_pair(
                np.array([[x_value, value]]), (col_idx - 1, col_idx)
            )
            return transformed_pair[0, 1].item()
        elif "x" in attribute:
            y_value: float = self.scene_data_df.iat[
                self.index_dict[(agent_id, scene_ts)], col_idx + 1
            ]

            transformed_pair: np.ndarray = self._transform_pair(
                np.array([[value, y_value]]), (col_idx, col_idx + 1)
            )
            return transformed_pair[0, 0].item()
        else:
            # The 0.0 and 0 here are just placeholders.
            transformed_pair: np.ndarray = self._transform_pair(
                np.array([[value, 0.0]]), (col_idx, 0)
            )
            return transformed_pair[0, 0].item()

    def get_raw_state(self, agent_id: str, scene_ts: int) -> np.ndarray:
        return (
            self.scene_data_df.iloc[
                self.index_dict[(agent_id, scene_ts)], : self._state_dim
            ]
            .to_numpy()
            .copy()
        )

    def get_state(self, agent_id: str, scene_ts: int) -> np.ndarray:
        state = self.get_raw_state(agent_id, scene_ts)

        return self._transform_state(state)

    def get_states(self, agent_ids: List[str], scene_ts: int) -> np.ndarray:
        row_idxs: List[int] = [
            self.index_dict[(agent_id, scene_ts)] for agent_id in agent_ids
        ]

        states = self.scene_data_df.iloc[row_idxs, : self._state_dim].to_numpy()

        return self._transform_state(states)

    def transform_data(self, **kwargs) -> None:
        if "shift_mean_to" in kwargs:
            # This standardizes the scene to be relative to the agent being predicted.
            self._transf_mean = kwargs["shift_mean_to"]

        if "rotate_by" in kwargs:
            # This rotates the scene so that the predicted agent's current
            # heading aligns with the x-axis.
            agent_heading: float = kwargs["rotate_by"]
            self._transf_rotmat: np.ndarray = np.array(
                [
                    [np.cos(agent_heading), -np.sin(agent_heading)],
                    [np.sin(agent_heading), np.cos(agent_heading)],
                ]
            )

        if "sincos_heading" in kwargs:
            self._sincos_heading = True
            self.obs_dim = self._state_dim + 1

    def reset_transforms(self) -> None:
        self._transf_mean: Optional[np.ndarray] = None
        self._transf_rotmat: Optional[np.ndarray] = None
        self._sincos_heading: bool = False

        self.obs_dim = self._state_dim

    def _transform_state(self, data: np.ndarray) -> np.ndarray:
        state: np.ndarray = data.copy()  # Don't want to alter the original data.

        if len(state.shape) == 1:
            state = state[np.newaxis, :]

        if self._transf_mean is not None:
            # Shift to zero mean, but leave heading
            # standardization for if rotation is also requested (below).
            state[..., :-1] -= self._transf_mean[:-1]

        if self._transf_rotmat is not None:
            state[..., self.pos_cols] = state[..., self.pos_cols] @ self._transf_rotmat
            state[..., self.vel_cols] = state[..., self.vel_cols] @ self._transf_rotmat
            state[..., self.acc_cols] = state[..., self.acc_cols] @ self._transf_rotmat
            state[..., -1] -= self._transf_mean[-1]

        if self._sincos_heading:
            sin_heading: np.ndarray = np.sin(state[..., [-1]])
            cos_heading: np.ndarray = np.cos(state[..., [-1]])
            state = np.concatenate([state[..., :-1], sin_heading, cos_heading], axis=-1)

        return state[0] if len(data.shape) == 1 else state

    def _transform_pair(
        self, data: np.ndarray, col_idxs: Tuple[int, int]
    ) -> np.ndarray:
        state: np.ndarray = data.copy()  # Don't want to alter the original data.

        if len(state.shape) == 1:
            state = state[np.newaxis, :]

        if self._transf_mean is not None:
            state -= self._transf_mean[col_idxs]

        if (
            self._transf_rotmat is not None
            and col_idxs[0] in self.pos_cols + self.vel_cols + self.acc_cols
        ):
            state = state @ self._transf_rotmat

        return state[0] if len(data.shape) == 1 else state

    def _upsample_data(
        self, new_index: pd.MultiIndex, upsample_dt_ratio: float, method: str
    ) -> pd.DataFrame:
        upsample_dt_factor: int = int(upsample_dt_ratio)

        interpolated_df: pd.DataFrame = pd.DataFrame(
            index=new_index, columns=self.scene_data_df.columns
        )
        interpolated_df = interpolated_df.astype(self.scene_data_df.dtypes.to_dict())

        scene_data: np.ndarray = self.scene_data_df.to_numpy()
        unwrapped_heading: np.ndarray = np.unwrap(self.scene_data_df["heading"])

        # Getting the data initially in the new df, making sure to unwrap angles above
        # in preparation for interpolation.
        scene_data_idxs: np.ndarray = np.nonzero(
            new_index.get_level_values("scene_ts") % upsample_dt_factor == 0
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

        return interpolated_df

    def _downsample_data(
        self, new_index: pd.MultiIndex, downsample_dt_ratio: float
    ) -> pd.DataFrame:
        downsample_dt_factor: int = int(downsample_dt_ratio)

        subsample_index: pd.MultiIndex = new_index.set_levels(
            new_index.levels[1] * downsample_dt_factor, level=1
        )

        subsampled_df: pd.DataFrame = self.scene_data_df.reindex(
            index=subsample_index
        ).set_index(new_index)

        return subsampled_df

    def interpolate_data(self, desired_dt: float, method: str = "linear") -> None:
        upsample_dt_ratio: float = self.scene.env_metadata.dt / desired_dt
        downsample_dt_ratio: float = desired_dt / self.scene.env_metadata.dt
        if not upsample_dt_ratio.is_integer() and not downsample_dt_ratio.is_integer():
            raise ValueError(
                f"{str(self.scene)}'s dt of {self.scene.dt}s "
                f"is not integer divisible by the desired dt {desired_dt}s."
            )

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

        if upsample_dt_ratio >= 1:
            interpolated_df = self._upsample_data(new_index, upsample_dt_ratio, method)
        elif downsample_dt_ratio >= 1:
            interpolated_df = self._downsample_data(new_index, downsample_dt_ratio)

        self.dt = desired_dt
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
                scene_ts - agent_history_df.shape[0] + 1, scene_ts
            )
        else:
            agent_extent_np = agent_history_df.iloc[:, self.extent_cols].to_numpy()

        return (
            self._transform_state(
                agent_history_df.iloc[:, : self._state_dim].to_numpy()
            ),
            agent_extent_np,
        )

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
            return np.zeros((0, self.obs_dim)), np.zeros((0, 3))

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
                scene_ts + 1, scene_ts + agent_future_df.shape[0]
            )
        else:
            agent_extent_np = agent_future_df.iloc[:, self.extent_cols].to_numpy()

        return (
            self._transform_state(
                agent_future_df.iloc[:, : self._state_dim].to_numpy()
            ),
            agent_extent_np,
        )

    def get_agents_history(
        self,
        scene_ts: int,
        agents: List[AgentMetadata],
        history_sec: Tuple[Optional[float], Optional[float]],
    ) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
        first_timesteps = np.array(
            [agent.first_timestep for agent in agents], dtype=int
        )
        if history_sec[1] is not None:
            max_history: int = floor(history_sec[1] / self.dt)
            first_timesteps = np.maximum(scene_ts - max_history, first_timesteps)

        first_index_incl: np.ndarray = np.array(
            [
                self.index_dict[(agent.name, first_timesteps[idx])]
                for idx, agent in enumerate(agents)
            ],
            dtype=int,
        )
        last_index_incl: np.ndarray = np.array(
            [self.index_dict[(agent.name, scene_ts)] for agent in agents], dtype=int
        )

        concat_idxs = arr_utils.vrange(first_index_incl, last_index_incl + 1)
        neighbor_data_df: pd.DataFrame = self.scene_data_df.iloc[concat_idxs, :]

        neighbor_history_lens_np = last_index_incl - first_index_incl + 1

        neighbor_histories_np = self._transform_state(
            neighbor_data_df.iloc[:, : self._state_dim].to_numpy()
        )
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
                    scene_ts - neighbor_history_lens_np[idx].item() + 1,
                    scene_ts,
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
        last_timesteps = np.array([agent.last_timestep for agent in agents], dtype=int)

        first_timesteps = np.minimum(scene_ts + 1, last_timesteps)

        if future_sec[1] is not None:
            max_future: int = floor(future_sec[1] / self.dt)
            last_timesteps = np.minimum(scene_ts + max_future, last_timesteps)

        first_index_incl: np.ndarray = np.array(
            [
                self.index_dict[(agent.name, first_timesteps[idx])]
                for idx, agent in enumerate(agents)
            ],
            dtype=int,
        )
        last_index_incl: np.ndarray = np.array(
            [
                self.index_dict[(agent.name, last_timesteps[idx])]
                for idx, agent in enumerate(agents)
            ],
            dtype=int,
        )

        concat_idxs = arr_utils.vrange(first_index_incl, last_index_incl + 1)
        neighbor_data_df: pd.DataFrame = self.scene_data_df.iloc[concat_idxs, :]

        neighbor_future_lens_np = last_index_incl - first_index_incl + 1

        neighbor_futures_np = self._transform_state(
            neighbor_data_df.iloc[:, : self._state_dim].to_numpy()
        )
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
                    scene_ts - neighbor_future_lens_np[idx].item() + 1,
                    scene_ts,
                )
                for idx, agent in enumerate(agents)
            ]

        return (
            neighbor_futures,
            neighbor_extents,
            neighbor_future_lens_np,
        )

    # TRAFFIC LIGHT INFO
    @staticmethod
    def _tls_data_file(scene_dt: float) -> str:
        return f"tls_data_dt{scene_dt:.2f}.feather"

    @staticmethod
    def save_traffic_light_data(
        traffic_light_status_data: pd.DataFrame,
        cache_path: Path,
        scene: Scene,
        dt: Optional[float] = None,
    ) -> None:
        """
        Assumes traffic_light_status_data is a MultiIndex dataframe with
        lane_connector_id and scene_ts as the indices, and has a column "status" with integer
        values for traffic status given by the TrafficLightStatus enum
        """
        scene_cache_dir: Path = DataFrameCache.scene_cache_dir(
            cache_path, scene.env_name, scene.name
        )
        scene_cache_dir.mkdir(parents=True, exist_ok=True)

        if dt is None:
            dt = scene.dt

        traffic_light_status_data.reset_index().to_feather(
            scene_cache_dir / DataFrameCache._tls_data_file(dt)
        )

    def is_traffic_light_data_cached(self, desired_dt: Optional[float] = None) -> bool:
        desired_dt = self.dt if desired_dt is None else desired_dt
        tls_data_path: Path = self.scene_dir / DataFrameCache._tls_data_file(desired_dt)
        return tls_data_path.exists()

    def get_traffic_light_status_dict(
        self, desired_dt: Optional[float] = None
    ) -> Dict[Tuple[int, int], TrafficLightStatus]:
        """
        Returns dict mapping Lane Id, scene_ts to traffic light status for the
        particular scene. If data doesn't exist for the current dt, interpolates and
        saves the interpolated data to disk for loading later.
        """
        desired_dt = self.dt if desired_dt is None else desired_dt

        tls_data_path: Path = self.scene_dir / DataFrameCache._tls_data_file(desired_dt)
        if not tls_data_path.exists():
            # Load the original dt traffic light data
            tls_orig_dt_df: pd.DataFrame = pd.read_feather(
                self.scene_dir
                / DataFrameCache._tls_data_file(self.scene.env_metadata.dt),
                use_threads=False,
            ).set_index(["lane_connector_id", "scene_ts"])

            # Interpolate it to the desired dt.
            tls_data_df = df_utils.interpolate_multi_index_df(
                tls_orig_dt_df, self.scene.env_metadata.dt, desired_dt, method="nearest"
            )

            # Save it for the future
            DataFrameCache.save_traffic_light_data(
                tls_data_df, self.path, self.scene, desired_dt
            )
        else:
            # Load the data with the desired dt.
            tls_data_df: pd.DataFrame = pd.read_feather(
                tls_data_path,
                use_threads=False,
            ).set_index(["lane_connector_id", "scene_ts"])

        # Return data as dict
        return {
            idx: TrafficLightStatus(v["status"]) for idx, v in tls_data_df.iterrows()
        }

    # MAPS
    @staticmethod
    def get_maps_path(cache_path: Path, env_name: str) -> Path:
        return cache_path / env_name / "maps"

    @staticmethod
    def are_maps_cached(cache_path: Path, env_name: str) -> bool:
        return DataFrameCache.get_maps_path(cache_path, env_name).exists()

    @staticmethod
    def get_map_paths(
        cache_path: Path, env_name: str, map_name: str, resolution: float
    ) -> Tuple[Path, Path, Path, Path, Path]:
        maps_path: Path = DataFrameCache.get_maps_path(cache_path, env_name)

        vector_map_path: Path = maps_path / f"{map_name}.pb"
        kdtrees_path: Path = maps_path / f"{map_name}_kdtrees.dill"
        raster_map_path: Path = maps_path / f"{map_name}_{resolution:.2f}px_m.zarr"
        raster_metadata_path: Path = maps_path / f"{map_name}_{resolution:.2f}px_m.dill"

        return (
            maps_path,
            vector_map_path,
            kdtrees_path,
            raster_map_path,
            raster_metadata_path,
        )

    @staticmethod
    def is_map_cached(
        cache_path: Path, env_name: str, map_name: str, resolution: float
    ) -> bool:
        (
            maps_path,
            vector_map_path,
            kdtrees_path,
            raster_map_path,
            raster_metadata_path,
        ) = DataFrameCache.get_map_paths(cache_path, env_name, map_name, resolution)
        return (
            maps_path.exists()
            and vector_map_path.exists()
            and kdtrees_path.exists()
            and raster_metadata_path.exists()
            and raster_map_path.exists()
        )

    @staticmethod
    def finalize_and_cache_map(
        cache_path: Path,
        vector_map: VectorMap,
        map_params: Dict[str, Any],
    ) -> None:
        raster_resolution: float = map_params["px_per_m"]

        (
            maps_path,
            vector_map_path,
            kdtrees_path,
            raster_map_path,
            raster_metadata_path,
        ) = DataFrameCache.get_map_paths(
            cache_path, vector_map.env_name, vector_map.map_name, raster_resolution
        )

        pbar_kwargs = {"position": 2, "leave": False}
        rasterized_map: RasterizedMap = raster_utils.rasterize_map(
            vector_map, raster_resolution, **pbar_kwargs
        )

        vector_map.compute_search_indices()

        # Ensuring the maps directory exists.
        maps_path.mkdir(parents=True, exist_ok=True)

        # Saving the vectorized map data.
        with open(vector_map_path, "wb") as f:
            f.write(vector_map.to_proto().SerializeToString())

        # Saving precomputed map element kdtrees.
        with open(kdtrees_path, "wb") as f:
            dill.dump(vector_map.search_kdtrees, f)

        # Saving the rasterized map data.
        zarr.save(raster_map_path, rasterized_map.data)

        # Saving the rasterized map metadata.
        with open(raster_metadata_path, "wb") as f:
            dill.dump(rasterized_map.metadata, f)

    def pad_map_patch(
        self,
        patch: np.ndarray,
        #                 top, bot, left, right
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

    def load_kdtrees(self) -> Dict[str, MapElementKDTree]:
        _, _, kdtrees_path, _, _ = DataFrameCache.get_map_paths(
            self.path, self.scene.env_name, self.scene.location, 0.0
        )

        with open(kdtrees_path, "rb") as f:
            kdtrees: Dict[str, MapElementKDTree] = dill.load(f)

        return kdtrees

    def get_kdtrees(self, load_only_once: bool = True):
        """Loads and returns the kdtrees dictionary from the cache file.

        Args:
            load_only_once (bool): store the kdtree dictionary in self so that we
                dont have to load it from the cache file more than once.
        """
        if self._kdtrees is None:
            kdtrees = self.load_kdtrees()
            if load_only_once:
                self._kdtrees = kdtrees

            return kdtrees

        else:
            return self._kdtrees

    def load_map_patch(
        self,
        world_x: float,
        world_y: float,
        desired_patch_size: int,
        resolution: float,
        offset_xy: Tuple[float, float],
        agent_heading: float,
        return_rgb: bool,
        rot_pad_factor: float = 1.0,
        no_map_val: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        (
            maps_path,
            _,
            _,
            raster_map_path,
            raster_metadata_path,
        ) = DataFrameCache.get_map_paths(
            self.path, self.scene.env_name, self.scene.location, resolution
        )
        if not maps_path.exists():
            # This dataset (or location) does not have any maps,
            # so we return an empty map.
            patch_size: int = ceil((rot_pad_factor * desired_patch_size) / 2) * 2

            return (
                np.full(
                    (1 if not return_rgb else 3, patch_size, patch_size),
                    fill_value=no_map_val,
                ),
                np.eye(3),
                False,
            )

        with open(raster_metadata_path, "rb") as f:
            map_info: RasterizedMapMetadata = dill.load(f)

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

        # This first size is how much of the map we
        # need to extract to match the requested metric size (meters x meters) of
        # the patch.
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

        # This is the size of the patch taking into account expansion to allow for
        # rotation to match the agent's heading. We also ensure the final size is
        # divisible by two so that the // 2 below does not chop any information off.
        data_with_rot_pad_size: int = ceil((rot_pad_factor * data_patch_size) / 2) * 2

        disk_data = zarr.open_array(raster_map_path, mode="r")

        map_x = round(map_x)
        map_y = round(map_y)

        # Half of the patch's side length.
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

        return data_patch, raster_from_world_tf, True
