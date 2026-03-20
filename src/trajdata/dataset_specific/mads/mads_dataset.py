# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import argparse
import glob
import os
import random
import traceback
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Type, cast

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from tqdm import tqdm

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
    SUPPORTED_DATA_SRCS,
    USE_CUBIC_INTERPOLATION,
    ObstacleClassV1,
    resolve_data_src,
)
from trajdata.dataset_specific.mads.tar_extractor import TarExtractor
from trajdata.dataset_specific.raw_dataset import RawDataset
from trajdata.dataset_specific.scene_records import MadsSceneRecord
from trajdata.maps import VectorMap
from trajdata.utils.parallel_utils import parallel_apply


class MADSDataset(RawDataset):
    def compute_metadata(self, env_name: str, data_dir: str) -> EnvMetadata:
        """Scan clips and build dataset-level metadata.

        Args:
            env_name: Environment name for trajdata metadata.
            data_dir: Root data directory containing clip folders or tar files.

        Returns:
            Constructed `EnvMetadata` for this dataset instance.
        """
        # Create scene splits
        dataset_parts: List[Tuple[str, ...]] = [("all", "train", "val")]
        self.data_dir = data_dir

        # List to hold file names without extensions
        expanded_data_dir = str(Path(data_dir).expanduser())
        clip_dir = dict()
        clip_start_micros = dict()
        clip_duration = dict()
        tar_extractor = TarExtractor()
        for subdir in os.listdir(expanded_data_dir):
            # Construct the full path using the expanded data_dir

            # support for the clipgt2.3.0 that is extracted by the prep_parquet_given_clipids.py
            clip_files = glob.glob(
                os.path.join(expanded_data_dir, subdir) + "/**/clip.parquet",
                recursive=True,
            )
            clip_file = clip_files[0] if clip_files else None

            # if not extracted in advance, we can parse the raw tar
            if clip_file is None:
                # Check for tarred version -- force clipgt.2.0.0, TODO: make it a config option
                tar_files = glob.glob(
                    os.path.join(expanded_data_dir, subdir, "**/clipgt.2.0.0*.tar"),
                    recursive=True,
                )
                if tar_files:
                    extracted_dir = tar_extractor.get_clip_dir(Path(tar_files[0]))
                    extracted_clip_file = os.path.join(extracted_dir, "clip.parquet")
                    if os.path.exists(extracted_clip_file):
                        clip_file = extracted_clip_file

            if clip_file is None:
                # Could not find any clip file
                continue

            # Skip clips missing required ego trajectory parquet to avoid worker crashes.
            clip_parent_dir = str(Path(clip_file).parent.absolute())
            ego_file = os.path.join(clip_parent_dir, "egomotion_estimate.parquet")
            if not os.path.exists(ego_file):
                continue

            df_meta = pd.read_parquet(clip_file)
            df_meta = mads_utils.df_expand_json(df_meta)

            clip_id = df_meta["key.clip_id"][0]
            clip_dir[clip_id] = clip_parent_dir
            t0 = df_meta["key.time_range.start_micros"][0]
            tf = df_meta["key.time_range.end_micros"][0]
            clip_start_micros[clip_id] = t0
            clip_duration[clip_id] = (tf - t0) / 1e6

        # Cleanup to free the shmem.
        tar_extractor.cleanup()
        self.clip_start_micros = clip_start_micros

        # get all clip ids
        self.clip_duration = clip_duration
        clip_items = list(clip_dir.items())
        random.Random(42).shuffle(clip_items)
        self.clip_dir = dict(clip_items)
        clip_ids = list(clip_dir.keys())
        all_clips = [clip_id for clip_id, _ in clip_items[:]]

        scene_split_map: Dict[str, str] = {}
        for clip_id in all_clips:
            scene_split_map[clip_id] = "all"

        env_metadata = EnvMetadata(
            name=env_name,
            data_dir=data_dir,
            dt=MADS_DT,
            parts=cast(List[Tuple[str]], dataset_parts),
            scene_split_map=scene_split_map,
            # The location names should match the map names used in
            # the unified data cache.
            map_locations=tuple(clip_ids),
        )

        # Define the output file path (env override supported).
        self.clips_w_wm = os.getenv(
            "MADS_CLIPS_W_WM_PATH", "./tmp/clips_w_wm.txt"
        )
        clips_out_dir = os.path.dirname(self.clips_w_wm) or "."
        os.makedirs(clips_out_dir, exist_ok=True)

        return env_metadata

    def load_dataset_obj(self, verbose: bool = False) -> None:
        """Mark dataset as loaded so scene discovery uses the object path.

        MADS reads per-scene parquet files on demand and does not need a heavy
        in-memory dataset object. However, `RawDataset.get_matching_scenes()`
        uses `self.dataset_obj is None` as the signal to load from cache.
        Setting a lightweight sentinel prevents first runs from trying to read
        `scenes_list.dill` before it is created.
        """
        self.dataset_obj = True

    def _get_matching_scenes_from_obj(
        self,
        scene_tag: SceneTag,
        scene_desc_contains: Optional[List[str]],
        env_cache: EnvCache,
    ) -> List[SceneMetadata]:
        """Resolve scenes from in-memory clip listing.

        Args:
            scene_tag: Scene tag filter.
            scene_desc_contains: Optional description filter.
            env_cache: Environment cache for scene-list persistence.

        Returns:
            Matching scene metadata entries.
        """
        all_scenes_list: List[MadsSceneRecord] = list()

        scenes_list: List[SceneMetadata] = list()
        for idx, (clip_id, _clip_dir) in enumerate(self.clip_dir.items()):
            scene_location = "clipGT"
            if clip_id not in self.metadata.scene_split_map:
                print("Alert!!!!! scene {} not in scene_split_map".format(clip_id))
                continue

            scene_split: str = self.metadata.scene_split_map[clip_id]
            scene_length: int = int(self.clip_duration[clip_id] / MADS_DT)

            if scene_length > 1:
                all_scenes_list.append(
                    MadsSceneRecord(
                        clip_id, scene_location, str(scene_length), scene_split, idx
                    )
                )

            if (scene_split in scene_tag) and scene_desc_contains is None:
                scene_metadata = SceneMetadata(
                    env_name=self.metadata.name,
                    name=clip_id,
                    dt=self.metadata.dt,
                    raw_data_idx=idx,
                )
                scenes_list.append(scene_metadata)

        self.cache_all_scenes_list(env_cache, cast(List[NamedTuple], all_scenes_list))
        return scenes_list

    def _get_matching_scenes_from_cache(
        self,
        scene_tag: SceneTag,
        scene_desc_contains: Optional[List[str]],
        env_cache: EnvCache,
    ) -> List[Scene]:
        """Resolve scenes from cached scene records.

        Args:
            scene_tag: Scene tag filter.
            scene_desc_contains: Optional description filter.
            env_cache: Environment cache containing scene records.

        Returns:
            Matching `Scene` entries.
        """

        all_scenes_list = cast(
            List[MadsSceneRecord], env_cache.load_env_scenes_list(self.name)
        )

        scenes_list: List[Scene] = []
        for scene_record in all_scenes_list:
            (
                scene_name,
                scene_location,
                scene_length,
                scene_split,
                data_idx,
            ) = scene_record

            if scene_split in scene_tag and scene_desc_contains is None:
                scene_metadata = Scene(
                    self.metadata,
                    scene_name,
                    scene_location,
                    scene_split,
                    int(scene_length),
                    data_idx,
                    None,  # This isn't used if everything is already cached.
                )
                scenes_list.append(scene_metadata)

        return scenes_list

    def get_scene(self, scene_info: SceneMetadata) -> Scene:
        """Create a trajdata `Scene` from metadata."""
        # Type hinting for scene_info is not working properly in python 3.10

        scene_name = scene_info.name
        data_idx = scene_info.raw_data_idx

        scene_location = scene_info.name
        scene_split: str = self.metadata.scene_split_map[scene_name]
        scene_length: int = int(self.clip_duration[scene_name] / MADS_DT) + 1

        return Scene(
            self.metadata,
            scene_name,
            scene_location,
            scene_split,
            scene_length,
            data_idx,
            None,
        )

    @staticmethod
    def get_ego_df_from_path(
        scene_path: str,
        scene_name: str,
        verbose: bool = False,
        data_src: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load and normalize ego/dynamic data into a time-aligned dataframe.

        Args:
            scene_path: Path to one clip directory.
            scene_name: Clip ID.
            verbose: Whether to print debug information.
            data_src: Optional dataset source override (e.g., v2 or pai).

        Returns:
            Normalized dataframe with one row per `(agent_id, scene_ts)`.
        """

        resolved_data_src = resolve_data_src(data_src)

        if resolved_data_src == "v2":
            obstacle_path = os.path.join(scene_path, "object_fused.parquet")
            if not os.path.exists(obstacle_path):
                if verbose:
                    print(
                        f"Skipping scene {scene_name} as it does not have obstacle data.",
                        flush=True,
                    )
                dynamic_df = pd.DataFrame()
            else:
                dynamic_df = pd.read_parquet(obstacle_path)
                try:
                    dynamic_df = dynamic_df[(dynamic_df["key.clip_id"] == scene_name)]
                    dynamic_df["object_fused.obstacle_class"] = dynamic_df[
                        "object_fused.obstacle_class"
                    ].apply(ObstacleClassV1.get_enum_name)
                    dynamic_df = mads_utils.add_quaternion_from_direction(
                        dynamic_df,
                        "object_fused.obstacle_direction.x",
                        "object_fused.obstacle_direction.y",
                        "object_fused.obstacle_direction.z",
                    )
                    dynamic_df["agent_id"] = dynamic_df[
                        "object_fused.obstacle_id"
                    ].astype(str)
                    dynamic_df["length"] = (
                        dynamic_df["object_fused.cuboid_3D_halfAxisXYZ.x"].values * 2
                    )
                    dynamic_df["width"] = (
                        dynamic_df["object_fused.cuboid_3D_halfAxisXYZ.y"].values * 2
                    )
                    dynamic_df["height"] = (
                        dynamic_df["object_fused.cuboid_3D_halfAxisXYZ.z"].values * 2
                    )
                except Exception:
                    if verbose:
                        print(
                            f"Skipping scene {scene_name} as it does not have obstacle data.",
                            flush=True,
                        )
                    dynamic_df = pd.DataFrame()

            # read ego motion data
            ego_df = pd.read_parquet(
                os.path.join(scene_path, f"egomotion_estimate.parquet")
            )
            ego_df = mads_utils.df_expand_json(ego_df)
            egomotion_key = "EgomotionEstimate"

        elif resolved_data_src == "pai":
            raise NotImplementedError(
                "DATA_SRC='pai' placeholder is defined but parsing is not implemented yet."
            )
        else:
            raise ValueError(
                f"{resolved_data_src} not supported in trajdata mads dataset"
            )

        assert ego_df[f"{egomotion_key}.name"].unique().size == 1
        ego_df = ego_df.assign(
            length=EGO_LENGTH,
            width=EGO_WIDTH,
            height=EGO_HEIGHT,
            type="automobile",
            agent_id="ego",
            source="manual",
        )
        ego_df["key.label_class_id"] = ego_df[f"{egomotion_key}.name"].iat[0]

        assert ego_df[f"{egomotion_key}.name"].unique().size == 1
        ego_df = ego_df.drop(columns=[f"{egomotion_key}.name"])

        # re-naming the fields
        ego_df.rename(
            columns={
                f"{egomotion_key}.location.x": "x",
                f"{egomotion_key}.location.y": "y",
                f"{egomotion_key}.location.z": "z",
                f"{egomotion_key}.orientation.x": "qx",
                f"{egomotion_key}.orientation.y": "qy",
                f"{egomotion_key}.orientation.z": "qz",
                f"{egomotion_key}.orientation.w": "qw",
            },
            inplace=True,
        )
        dynamic_df.rename(
            columns={
                "Obstacle.center.x": "x",
                "Obstacle.center.y": "y",
                "Obstacle.center.z": "z",
                "object_fused.cuboid_3D_center.x": "x",
                "object_fused.cuboid_3D_center.y": "y",
                "object_fused.cuboid_3D_center.z": "z",
                "Obstacle.orientation.x": "qx",
                "Obstacle.orientation.y": "qy",
                "Obstacle.orientation.z": "qz",
                "Obstacle.orientation.w": "qw",
                "Obstacle.size.x": "length",
                "Obstacle.size.y": "width",
                "Obstacle.size.z": "height",
                "Obstacle.size.dimX": "length",
                "Obstacle.size.dimY": "width",
                "Obstacle.size.dimZ": "height",
                "Obstacle.category": "type",
                "object_fused.obstacle_class": "type",
            },
            inplace=True,
        )

        # timestamp
        t0 = ego_df["key.timestamp_micros"].iat[0]
        tf = ego_df["key.timestamp_micros"].iat[-1]
        if verbose:
            print("dynamic_df.empty", dynamic_df.empty)

        # Only select relevant dynamic data
        dynamic_df = pd.concat([ego_df, dynamic_df])
        dynamic_df = dynamic_df[(dynamic_df["key.timestamp_micros"] <= tf)]
        dynamic_df["rel_time_seconds"] = (dynamic_df["key.timestamp_micros"] - t0) / 1e6

        interpolated_dfs: List[pd.DataFrame] = []
        for group_name, group_df_raw in dynamic_df.groupby(
            ["key.clip_id", "key.label_class_id", "agent_id"]
        ):
            group_df: pd.DataFrame = cast(pd.DataFrame, group_df_raw)
            group_df = group_df.sort_values(by=["rel_time_seconds"])
            duplicated = group_df.duplicated(subset=["rel_time_seconds"])
            duplicated_mask = np.asarray(duplicated, dtype=bool)

            if duplicated_mask.sum() > 0:
                if verbose:
                    print(f"Duplicated timestamps found for agent: {group_name}")
                group_df = group_df.loc[~duplicated_mask]

            min_time = group_df["rel_time_seconds"].min()
            min_step = int(np.ceil(min_time / MADS_DT))
            max_time = group_df["rel_time_seconds"].max()
            max_step = int(np.floor(max_time / MADS_DT))
            target_steps = np.arange(min_step, max_step + 1)
            target_times = target_steps * MADS_DT

            if max_step - min_step + 1 < MIN_FRAMES:
                continue

            def _interp(col_name):
                x = group_df["rel_time_seconds"]
                y = group_df[col_name]
                if USE_CUBIC_INTERPOLATION:
                    return CubicSpline(x, y)(target_times)
                return np.interp(target_times, x, y)

            # [N, 4]
            quats_tensor = np.stack(
                [group_df["qx"], group_df["qy"], group_df["qz"], group_df["qw"],],
                axis=1,
            )
            # Takes in scalar-last quaternion (x, y, z, w)
            r = R.from_quat(quats_tensor)
            slerp = Slerp(group_df["rel_time_seconds"], r)
            interp_r = slerp(target_times)
            headings = interp_r.as_euler("zyx", degrees=False)[:, 0]
            # Scalar-last
            # interp_quats = interp_r.as_quat()

            df = pd.DataFrame(
                {
                    "key.clip_id": group_name[0],
                    "key.label_class_id": group_name[1],
                    "agent_id": group_name[2],
                    "scene_ts": target_steps,
                    "rel_time_seconds": target_times,
                    "x": _interp("x"),
                    "y": _interp("y"),
                    "z": _interp("z"),
                    # "qx": interp_quats[:, 0],
                    # "qy": interp_quats[:, 1],
                    # "qz": interp_quats[:, 2],
                    # "qw": interp_quats[:, 3],
                    "heading": headings,
                    # We interpolate this as this might change!
                    # In particular, I found this to change for manual labels.
                    "length": _interp("length"),
                    "width": _interp("width"),
                    "height": _interp("height"),
                    # "length": group_df["length"].iat[0],
                    # "width": group_df["width"].iat[0],
                    # "height": group_df["height"].iat[0],
                    "type": group_df["type"].iat[0],
                    "source": group_df["source"].iat[0],
                }
            )

            df["vx"] = df["x"].diff() / MADS_DT
            df["vy"] = df["y"].diff() / MADS_DT

            # Calculate ego accelerations 'ax' and 'ay'
            df["ax"] = df["vx"].diff() / MADS_DT
            df["ay"] = df["vy"].diff() / MADS_DT

            # Replace infinity with nan for later nan handling
            df["ax"] = df["ax"].replace([np.inf, -np.inf], np.nan)
            df["ay"] = df["ay"].replace([np.inf, -np.inf], np.nan)

            # The first row of ax and ay is NaN, fill in values where NaN exists
            df["vx"] = df["vx"].bfill().ffill()
            df["vy"] = df["vy"].bfill().ffill()
            df["ax"] = df["ax"].bfill().ffill()
            df["ay"] = df["ay"].bfill().ffill()

            interpolated_dfs.append(df)
        interpolated_df: pd.DataFrame = pd.concat(interpolated_dfs).reset_index(drop=True)
        assert int(interpolated_df.duplicated(subset=["scene_ts", "agent_id"]).sum()) == 0

        T = (tf - t0) / (1e6 * MADS_DT)
        scene_ts_series: pd.Series = cast(pd.Series, interpolated_df.loc[:, "scene_ts"])
        valid_scene_ts_mask: pd.Series = cast(
            pd.Series, (scene_ts_series >= 0) & (scene_ts_series <= T)
        )
        valid_scene_ts_mask_np = np.asarray(valid_scene_ts_mask, dtype=bool)
        interpolated_df = interpolated_df.loc[valid_scene_ts_mask_np]

        # Sort by distance to ego
        ego_start = interpolated_df.query("agent_id == 'ego' and scene_ts == 0")
        ego_x = ego_start["x"].iat[0]
        ego_y = ego_start["y"].iat[0]

        unique_distances = set()

        def get_group_distance_to_ego(group_df: pd.DataFrame) -> pd.DataFrame:
            min_ts = group_df["scene_ts"].min()
            agent_start = group_df.query(f"scene_ts == {min_ts}")
            agent_x = agent_start["x"].iat[0]
            agent_y = agent_start["y"].iat[0]
            distance_to_ego = np.sqrt((ego_x - agent_x) ** 2 + (ego_y - agent_y) ** 2)
            assert distance_to_ego not in unique_distances
            unique_distances.add(distance_to_ego)
            group_df["distance_to_ego"] = distance_to_ego
            return group_df

        sorted_df: pd.DataFrame = cast(
            pd.DataFrame,
            cast(Any, interpolated_df.groupby("agent_id")).apply(
                get_group_distance_to_ego, include_groups=False
            ),
        )
        sorted_df = (
            sorted_df.sort_values(by=["distance_to_ego", "agent_id", "scene_ts"])
            .reset_index()
            .drop(columns=["level_1"])
        )

        # Filter out agents that are too close to each other
        # Strategy: For simplicity and speed we only compare the xy locations of agents
        # when they are first seen. This might ofc missing cases when the agent moves
        # and the 'ghost' object appears later.
        # We start by adding all agents with gt labels. Then, we iterate over the rest
        # of the agents and either:
        # - accept them and add their first seen location to `first_states`
        # - reject them and add them to `agents_to_remove`
        def get_row_first_seen(df: pd.DataFrame) -> pd.DataFrame:
            return (
                cast(Any, df.groupby("agent_id", sort=False)).apply(
                    lambda gdf: gdf.iloc[0], include_groups=False
                )
                .reset_index()
            )

        first_states = get_row_first_seen(sorted_df.query("source == 'manual'"))

        agents_to_remove: set[str] = set()
        # Add agents one by one if they don't overlap
        for agent_id, group_df in sorted_df.query("source != 'manual'").groupby(
            "agent_id", sort=False
        ):
            current_agent_first_state = group_df.iloc[0]
            first_states["distance_to_current_agent"] = np.sqrt(
                (first_states["x"] - current_agent_first_state.x) ** 2
                + (first_states["y"] - current_agent_first_state.y) ** 2
            )
            closest_state = first_states.sort_values(
                by=["distance_to_current_agent"]
            ).iloc[0]
            # Only conservative filtering based on dimensions:
            if (
                min(closest_state.width, closest_state.height)
                > closest_state.distance_to_current_agent
            ):
                agents_to_remove.add(agent_id)
            else:
                first_states = pd.concat([first_states, get_row_first_seen(group_df)])

        agent_id_series = cast(pd.Series, sorted_df["agent_id"])
        keep_mask = ~cast(Any, agent_id_series).isin(list(agents_to_remove))
        sorted_df = cast(pd.DataFrame, sorted_df[keep_mask])

        return sorted_df

    def get_agent_info(
        self, scene: Scene, cache_path: Path, cache_class: Type[SceneCache]
    ) -> Tuple[List[AgentMetadata], List[List[AgentMetadata]]]:
        sorted_df = self.get_ego_df_from_path(self.clip_dir[scene.name], scene.name)

        contain_obstacles: bool = False
        agent_list: List[AgentMetadata] = []
        agent_presence: List[List[AgentMetadata]] = [
            [] for _ in range(scene.length_timesteps)
        ]
        agents_to_remove = []
        for agent_id, frames in sorted_df.groupby("agent_id", sort=False)[
            ["scene_ts", "type", "length", "width", "height"]
        ]:
            all_frame_ids = frames["scene_ts"]

            start_frame: int = all_frame_ids.iat[0]
            last_frame: int = all_frame_ids.iat[-1]

            # XW: added for v0tar data, sometimes the clip duration is short
            last_frame = min(last_frame, scene.length_timesteps)

            if agent_id != "ego":
                contain_obstacles = True

            agent_length = (
                frames["length"].iloc[0][0]
                if isinstance(frames["length"].iloc[0], list)
                else frames["length"].iloc[0]
            )
            agent_width = (
                frames["width"].iloc[0][0]
                if isinstance(frames["width"].iloc[0], list)
                else frames["width"].iloc[0]
            )
            agent_height = (
                frames["height"].iloc[0][0]
                if isinstance(frames["height"].iloc[0], list)
                else frames["height"].iloc[0]
            )

            agent_metadata = AgentMetadata(
                name=agent_id,
                agent_type=mads_utils.mads_type_to_unified_type(frames["type"].iloc[0]),
                first_timestep=start_frame,
                last_timestep=last_frame,
                extent=FixedExtent(
                    length=agent_length, width=agent_width, height=agent_height
                ),
            )

            agent_list.append(agent_metadata)
            for frame in range(
                agent_metadata.first_timestep, agent_metadata.last_timestep
            ):
                agent_presence[frame].append(agent_metadata)

        agent_id_series = cast(pd.Series, sorted_df["agent_id"])
        keep_mask = ~cast(Any, agent_id_series).isin(list(agents_to_remove))
        sorted_df = cast(pd.DataFrame, sorted_df[keep_mask])
        sorted_df.set_index(["agent_id", "scene_ts"], inplace=True)

        cache_class.save_agent_data(
            sorted_df, cache_path, scene,
        )

        # also save clip ids that contain WM data
        if contain_obstacles:
            with open(self.clips_w_wm, "a", encoding="utf-8") as f:
                f.write(f"{scene.name}\n")

        return agent_list, agent_presence

    def cache_map(
        self,
        map_name: str,
        cache_path: Path,
        map_cache_class: Type[SceneCache],
        map_params: Dict[str, Any],
        verbose: bool = False,
    ) -> None:
        """Cache one map into trajdata map cache if not already present."""

        save_file = os.path.join(
            cache_path, self.metadata.name, "maps", f"{map_name}_4.00px_m.dill"
        )
        if os.path.exists(save_file):
            if verbose:
                print(f"Skipping {map_name} Map", flush=True)
        elif resolve_data_src() == "v2":
            vector_map = VectorMap(map_id=f"{self.name}:{map_name}")
            try:
                mads_utils.populate_vector_map(vector_map, self.clip_dir[map_name])
                map_cache_class.finalize_and_cache_map(
                    cache_path, vector_map, map_params
                )
            except FileNotFoundError:
                # Some clips do not include lane parquet files; skip quietly.
                if verbose:
                    print(
                        f"[MapCache] Missing lane parquet for {map_name}, skipping.",
                        flush=True,
                    )
                return
            except Exception:
                print(
                    f"[MapCache] Failed to cache map {map_name}, skipping.", flush=True
                )
                traceback.print_exc()
                return
        elif resolve_data_src() == "pai":
            raise NotImplementedError(
                "DATA_SRC='pai' placeholder is defined but map caching is not implemented yet."
            )
        else:
            raise ValueError("not supported")

    def cache_maps(
        self,
        cache_path: Path,
        map_cache_class: Type[SceneCache],
        map_params: Dict[str, Any],
        resume: bool = True,
    ) -> None:
        """Cache maps for all clips, optionally skipping already cached maps."""

        # select the ones that are not finished
        if resume:
            clip_dir_need_map = []
            for clip_id in self.clip_dir.keys():
                map_file = os.path.join(
                    cache_path, self.metadata.name, "maps", f"{clip_id}_4.00px_m.dill"
                )
                if not os.path.exists(map_file):
                    clip_dir_need_map.append(clip_id)
            clip_list = clip_dir_need_map
        else:
            clip_list = self.clip_dir.keys()

        num_workers: int = map_params.get("num_workers", 0)
        if num_workers > 1:
            parallel_apply(
                partial(
                    self.cache_map,
                    cache_path=cache_path,
                    map_cache_class=map_cache_class,
                    map_params=map_params,
                ),
                clip_list,
                num_workers=num_workers,
            )

        else:
            for map_name in tqdm(
                clip_list,
                desc=f"Caching {self.name} Maps at {map_params['px_per_m']:.2f} px/m",
                position=0,
            ):
                self.cache_map(map_name, cache_path, map_cache_class, map_params)

        print(f"Caching Map finished", flush=True)



def _debug_dump_scene_df(data_src: Optional[str] = None) -> None:
    """Debug helper to inspect one scene dataframe when run as a script."""
    scene_path = 'path/to/source/data'
    scene_name = "762e063d-6eb9-43ae-959c-e53af10b53f9"
    scene_path = os.path.join(scene_path, scene_name)
    ego_df: pd.DataFrame = MADSDataset.get_ego_df_from_path(
        scene_path, scene_name, verbose=True, data_src=data_src
    )

    # Display basic information
    print("\n--- DataFrame Shape (rows, columns) ---")
    print(ego_df.shape)

    print("\n--- Column Names ---")
    print(ego_df.columns.tolist())

    print("\n--- First 5 Rows ---")
    print(ego_df.head())

    print("\n--- DataFrame Info ---")
    print(ego_df.info())

    # Identify object columns
    include_dtypes = cast(Any, ["object", "int64", "float64"])
    ego_df_any = cast(Any, ego_df)
    selected_df: pd.DataFrame = cast(pd.DataFrame, ego_df_any.select_dtypes(include=include_dtypes))
    object_columns: List[str] = cast(List[str], [str(c) for c in list(selected_df.columns)])
    print("\nColumns with object dtype:", object_columns)

    # Analyze each object column
    for col in object_columns:
        print(f"\nAnalyzing column: {col}")

        # Get unique types in the column
        unique_types = ego_df[col].map(type).unique()
        print("Unique data types:", unique_types)

        # Display a few sample values
        sample_values = ego_df[col].dropna().sample(min(5, len(ego_df)), random_state=42)
        print("Sample values:", sample_values.tolist())

    # Show descriptive statistics
    print("\n--- Descriptive Statistics ---")
    print(ego_df.describe(include="all"))


if __name__ == "__main__":  # pyright: ignore[reportUnreachableCode]
    parser = argparse.ArgumentParser(description="Inspect MADS dataframe parsing")
    parser.add_argument(
        "--data-src",
        type=str,
        default=None,
        choices=sorted(SUPPORTED_DATA_SRCS),
        help="Override data source (default: env MADS_DATA_SRC or constant DATA_SRC).",
    )
    args = parser.parse_args()
    _debug_dump_scene_df(data_src=args.data_src)
