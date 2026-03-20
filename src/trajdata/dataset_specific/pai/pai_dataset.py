# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import math
import os
import random
import zipfile
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Type, cast

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from trajdata.caching import EnvCache, SceneCache
from trajdata.data_structures.agent import AgentMetadata, FixedExtent
from trajdata.data_structures.environment import EnvMetadata
from trajdata.data_structures.scene_metadata import Scene, SceneMetadata
from trajdata.data_structures.scene_tag import SceneTag
from trajdata.dataset_specific.mads import mads_utils
from trajdata.dataset_specific.mads.constant import (
    EGO_HEIGHT,
    EGO_LENGTH,
    EGO_WIDTH,
    MADS_DT,
    MIN_FRAMES,
)
from trajdata.dataset_specific.raw_dataset import RawDataset
from trajdata.dataset_specific.scene_records import PAISceneRecord

_ZIP_MEMBER_SEP = "::"

class PAIDataset(RawDataset):
    def compute_metadata(self, env_name: str, data_dir: str) -> EnvMetadata:
        dataset_parts: List[Tuple[str, ...]] = [("all", "train", "val")]

        self.data_dir = str(Path(data_dir).expanduser())
        egomotion_dir = Path(self.data_dir) / "labels" / "egomotion"
        if not egomotion_dir.exists():
            raise FileNotFoundError(
                f"Expected PAI egomotion directory at {egomotion_dir}."
            )

        clip_dir: Dict[str, str] = {}
        clip_duration: Dict[str, float] = {}
        clip_start_micros: Dict[str, int] = {}

        for zip_path in sorted(egomotion_dir.glob("*.zip")):
            with zipfile.ZipFile(zip_path, "r") as zf:
                for member in zf.namelist():
                    if not member.endswith(".parquet"):
                        continue

                    file_name = Path(member).name
                    if not file_name:
                        continue

                    clip_id = file_name.split(".")[0]
                    if not clip_id or clip_id in clip_dir:
                        continue

                    clip_dir[clip_id] = f"{zip_path}{_ZIP_MEMBER_SEP}{member}"
                    clip_start_micros[clip_id] = 0
                    clip_duration[clip_id] = 20.0

        if not clip_dir:
            raise FileNotFoundError(
                f"No egomotion parquet files were found under {egomotion_dir}."
            )

        self.clip_start_micros = clip_start_micros
        self.clip_duration = clip_duration

        clip_items = list(clip_dir.items())
        random.shuffle(clip_items)
        self.clip_dir = dict(clip_items)

        all_clips = [clip_id for clip_id, _ in clip_items]
        scene_split_map: Dict[str, str] = {clip_id: "all" for clip_id in all_clips}

        return EnvMetadata(
            name=env_name,
            data_dir=self.data_dir,
            dt=MADS_DT,
            parts=cast(List[Tuple[str]], dataset_parts),
            scene_split_map=scene_split_map,
            map_locations=tuple(all_clips),
        )

    def load_dataset_obj(self, verbose: bool = False) -> None:
        pass

    def _get_matching_scenes_from_obj(
        self,
        scene_tag: SceneTag,
        scene_desc_contains: Optional[List[str]],
        env_cache: EnvCache,
    ) -> List[SceneMetadata]:
        all_scenes_list: List[PAISceneRecord] = []
        scenes_list: List[SceneMetadata] = []

        for idx, (clip_id, _clip_path) in enumerate(self.clip_dir.items()):
            scene_split = self.metadata.scene_split_map[clip_id]
            scene_length: int = int(self.clip_duration[clip_id] / MADS_DT)

            if scene_length > 1:
                all_scenes_list.append(
                    PAISceneRecord(clip_id, "pai", str(scene_length), scene_split, idx)
                )

            if (scene_split in scene_tag) and scene_desc_contains is None:
                scenes_list.append(
                    SceneMetadata(
                        env_name=self.metadata.name,
                        name=clip_id,
                        dt=self.metadata.dt,
                        raw_data_idx=idx,
                    )
                )

        self.cache_all_scenes_list(env_cache, cast(List[NamedTuple], all_scenes_list))
        return scenes_list

    def _get_matching_scenes_from_cache(
        self,
        scene_tag: SceneTag,
        scene_desc_contains: Optional[List[str]],
        env_cache: EnvCache,
    ) -> List[Scene]:
        all_scenes_list = cast(
            List[PAISceneRecord], env_cache.load_env_scenes_list(self.name)
        )

        scenes_list: List[Scene] = []
        for scene_record in all_scenes_list:
            scene_name, scene_location, scene_length, scene_split, data_idx = scene_record

            if scene_split in scene_tag and scene_desc_contains is None:
                scenes_list.append(
                    Scene(
                        self.metadata,
                        scene_name,
                        scene_location,
                        scene_split,
                        int(scene_length),
                        data_idx,
                        None,
                    )
                )

        return scenes_list

    def get_scene(self, scene_info: SceneMetadata) -> Scene:
        scene_name = scene_info.name
        data_idx = scene_info.raw_data_idx
        scene_split: str = self.metadata.scene_split_map[scene_name]
        scene_length: int = int(self.clip_duration[scene_name] / MADS_DT) + 1

        return Scene(
            self.metadata,
            scene_name,
            "pai",
            scene_split,
            scene_length,
            data_idx,
            None,
        )

    @staticmethod
    def _load_ego_df(scene_path: str) -> pd.DataFrame:
        if _ZIP_MEMBER_SEP in scene_path:
            zip_path, member = scene_path.split(_ZIP_MEMBER_SEP, maxsplit=1)
            with zipfile.ZipFile(zip_path, "r") as zf:
                with zf.open(member, "r") as f:
                    return pd.read_parquet(f)

        if os.path.isdir(scene_path):
            estimated = os.path.join(scene_path, "egomotion_estimate.parquet")
            if os.path.exists(estimated):
                return pd.read_parquet(estimated)

            egomotion = os.path.join(scene_path, "egomotion.parquet")
            if os.path.exists(egomotion):
                return pd.read_parquet(egomotion)

        return pd.read_parquet(scene_path)

    @staticmethod
    def get_df_from_path(
        scene_path: str,
        scene_name: str,
        verbose: bool = False,
    ) -> pd.DataFrame:
        ego_df = PAIDataset._load_ego_df(scene_path)
        ego_df = mads_utils.df_expand_json(ego_df)

        timestamp_col = "timestamp"
        if timestamp_col not in ego_df.columns:
            timestamp_col = "key.timestamp_micros"

        if timestamp_col not in ego_df.columns:
            raise KeyError(
                f"Expected timestamp column in egomotion parquet for scene {scene_name}."
            )

        if "key.clip_id" not in ego_df.columns:
            ego_df["key.clip_id"] = scene_name

        col_map = {
            "x": "x",
            "y": "y",
            "z": "z",
            "qx": "qx",
            "qy": "qy",
            "qz": "qz",
            "qw": "qw",
        }
        prefixed_map = {
            "EgomotionEstimate.location.x": "x",
            "EgomotionEstimate.location.y": "y",
            "EgomotionEstimate.location.z": "z",
            "EgomotionEstimate.orientation.x": "qx",
            "EgomotionEstimate.orientation.y": "qy",
            "EgomotionEstimate.orientation.z": "qz",
            "EgomotionEstimate.orientation.w": "qw",
        }

        if all(key in ego_df.columns for key in col_map.keys()):
            normalized = ego_df.rename(columns=col_map).copy()
        elif all(key in ego_df.columns for key in prefixed_map.keys()):
            normalized = ego_df.rename(columns=prefixed_map).copy()
        else:
            missing = [
                key
                for key in [
                    "x",
                    "y",
                    "z",
                    "qx",
                    "qy",
                    "qz",
                    "qw",
                    "EgomotionEstimate.location.x",
                    "EgomotionEstimate.location.y",
                    "EgomotionEstimate.location.z",
                    "EgomotionEstimate.orientation.x",
                    "EgomotionEstimate.orientation.y",
                    "EgomotionEstimate.orientation.z",
                    "EgomotionEstimate.orientation.w",
                ]
                if key not in ego_df.columns
            ]
            raise KeyError(
                f"Could not find expected ego pose columns for scene {scene_name}. Missing: {missing[:6]}"
            )

        normalized = normalized.sort_values(by=[timestamp_col]).drop_duplicates(
            subset=[timestamp_col], keep="first"
        )

        t0 = int(normalized[timestamp_col].iat[0])
        tf = int(normalized[timestamp_col].iat[-1])
        normalized["key.timestamp_micros"] = normalized[timestamp_col]
        normalized["rel_time_seconds"] = (normalized["key.timestamp_micros"] - t0) / 1e6

        min_time = float(normalized["rel_time_seconds"].min())
        max_time = float(normalized["rel_time_seconds"].max())
        min_step = int(math.ceil(min_time / MADS_DT))
        max_step = int(math.floor(max_time / MADS_DT))

        target_steps = np.arange(min_step, max_step + 1)
        target_times = target_steps * MADS_DT

        if target_steps.size < MIN_FRAMES:
            raise ValueError(
                f"Scene {scene_name} is too short after interpolation ({target_steps.size} < {MIN_FRAMES})."
            )

        def _interp(col_name: str) -> np.ndarray:
            return np.interp(
                target_times,
                normalized["rel_time_seconds"].to_numpy(),
                normalized[col_name].to_numpy(),
            )

        quats_tensor = np.stack(
            [
                normalized["qx"].to_numpy(),
                normalized["qy"].to_numpy(),
                normalized["qz"].to_numpy(),
                normalized["qw"].to_numpy(),
            ],
            axis=1,
        )
        interp_r = Slerp(normalized["rel_time_seconds"], R.from_quat(quats_tensor))(
            target_times
        )
        headings = interp_r.as_euler("zyx", degrees=False)[:, 0]

        df = pd.DataFrame(
            {
                "key.clip_id": scene_name,
                "key.label_class_id": "ego",
                "agent_id": "ego",
                "scene_ts": target_steps,
                "rel_time_seconds": target_times,
                "x": _interp("x"),
                "y": _interp("y"),
                "z": _interp("z"),
                "heading": headings,
                "length": EGO_LENGTH,
                "width": EGO_WIDTH,
                "height": EGO_HEIGHT,
                "type": "automobile",
                "source": "manual",
            }
        )

        if "vx" in normalized.columns and "vy" in normalized.columns:
            df["vx"] = _interp("vx")
            df["vy"] = _interp("vy")
        else:
            df["vx"] = df["x"].diff() / MADS_DT
            df["vy"] = df["y"].diff() / MADS_DT

        if "ax" in normalized.columns and "ay" in normalized.columns:
            df["ax"] = _interp("ax")
            df["ay"] = _interp("ay")
        else:
            df["ax"] = df["vx"].diff() / MADS_DT
            df["ay"] = df["vy"].diff() / MADS_DT

        df["vx"] = df["vx"].replace([np.inf, -np.inf], np.nan).bfill().ffill()
        df["vy"] = df["vy"].replace([np.inf, -np.inf], np.nan).bfill().ffill()
        df["ax"] = df["ax"].replace([np.inf, -np.inf], np.nan).bfill().ffill()
        df["ay"] = df["ay"].replace([np.inf, -np.inf], np.nan).bfill().ffill()

        t_horizon = (tf - t0) / (1e6 * MADS_DT)
        return cast(
            pd.DataFrame,
            df[(df["scene_ts"] >= 0) & (df["scene_ts"] <= t_horizon)].reset_index(
                drop=True
            ),
        )

    def get_agent_info(
        self, scene: Scene, cache_path: Path, cache_class: Type[SceneCache]
    ) -> Tuple[List[AgentMetadata], List[List[AgentMetadata]]]:
        ego_df = self.get_df_from_path(self.clip_dir[scene.name], scene.name)
        ego_df.set_index(["agent_id", "scene_ts"], inplace=True)

        ego_metadata = AgentMetadata(
            name="ego",
            agent_type=mads_utils.mads_type_to_unified_type("automobile"),
            first_timestep=0,
            last_timestep=min(scene.length_timesteps, int(ego_df.index.get_level_values(1).max())),
            extent=FixedExtent(length=EGO_LENGTH, width=EGO_WIDTH, height=EGO_HEIGHT),
        )

        agent_presence: List[List[AgentMetadata]] = [
            [] for _ in range(scene.length_timesteps)
        ]
        for frame in range(ego_metadata.first_timestep, ego_metadata.last_timestep):
            agent_presence[frame].append(ego_metadata)

        cache_class.save_agent_data(ego_df, cache_path, scene)
        return [ego_metadata], agent_presence

    def cache_map(
        self,
        map_name: str,
        cache_path: Path,
        map_cache_class: Type[SceneCache],
        map_params: Dict[str, Any],
        verbose: bool = False,
    ) -> None:
        return

    def cache_maps(
        self,
        cache_path: Path,
        map_cache_class: Type[SceneCache],
        map_params: Dict[str, Any],
        resume: bool = True,
    ) -> None:
        return
