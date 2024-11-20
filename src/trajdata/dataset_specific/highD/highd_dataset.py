import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Tuple, Type
from trajdata.dataset_specific.raw_dataset import RawDataset
from trajdata.data_structures.environment import EnvMetadata
from trajdata.data_structures import (
    AgentMetadata,
    EnvMetadata,
    Scene,
    SceneMetadata,
    SceneTag,
)
from trajdata.caching import EnvCache, SceneCache
from trajdata.dataset_specific.scene_records import HighDRecord
from trajdata.caching.df_cache import STATE_COLS, EXTENT_COLS
from trajdata.data_structures.agent import (
    AgentType,
    VariableExtent,
)
from trajdata.maps import RasterizedMapMetadata
from tqdm import tqdm
import cv2
import math


HIGHD_DT: Final[float] = 0.04
HIGHD_NUM_SCENES: Final[int] = 60
HIGHD_ENV_NAME: Final[str] = "highD"
HIGHD_SPLIT_NAME: Final[str] = "all"
# Scailing factor for the HighD raster map
# https://github.com/RobertKrajewski/highD-dataset/blob/master/Python/src/visualization/visualize_frame.py#L151-L152
HIGHD_PX_PER_M: Final[float] = 0.40424


class HighDDataset(RawDataset):
    def compute_metadata(self, env_name: str, data_dir: str) -> EnvMetadata:
        if env_name != HIGHD_ENV_NAME:
            raise ValueError(f"Invalid environment name: {env_name}")
        dataset_parts = [(HIGHD_SPLIT_NAME,)]
        scene_split_map = {
            str(scene_id): HIGHD_SPLIT_NAME
            for scene_id in range(1, HIGHD_NUM_SCENES + 1)
        }
        return EnvMetadata(
            name=env_name,
            data_dir=data_dir,
            dt=HIGHD_DT,
            parts=dataset_parts,
            scene_split_map=scene_split_map,
        )

    def load_dataset_obj(self, verbose: bool = False) -> None:
        if verbose:
            print(f"Loading {self.name} dataset...", flush=True)
        self.dataset_obj: Dict[int, Dict[str, Any]] = dict()
        for scene_id in tqdm(range(1, HIGHD_NUM_SCENES + 1)):
            raw_data_idx = scene_id - 1
            scene_id_str = str(scene_id).zfill(2)
            tracks_metadata = pd.read_csv(
                Path(self.metadata.data_dir) / f"{scene_id_str}_tracksMeta.csv"
            )
            tracks_metadata["id"] = tracks_metadata["id"].astype(str)
            tracks_data = pd.read_csv(
                Path(self.metadata.data_dir) / f"{scene_id_str}_tracks.csv"
            )
            tracks_data["id"] = tracks_data["id"].astype(str)
            tracks_data = tracks_data.merge(
                tracks_metadata[["id", "numFrames"]], on="id"
            )
            tracks_metadata.set_index("id", inplace=True)
            tracks_data = tracks_data[tracks_data["numFrames"] > 1].reset_index(
                drop=True
            )
            tracks_data["z"] = np.zeros_like(tracks_data["x"])
            # Regarding width -> length and height -> width plz see
            # https://levelxdata.com/wp-content/uploads/2023/10/highD-Format.pdf
            # Track Meta Information
            tracks_data.rename(
                columns={
                    "frame": "scene_ts",
                    "id": "agent_id",
                    "width": "length",
                    "height": "width",
                    "xVelocity": "vx",
                    "yVelocity": "vy",
                    "xAcceleration": "ax",
                    "yAcceleration": "ay",
                },
                inplace=True,
            )
            # Originally in the data:
            # The x position of the upper left corner of the vehicle's bounding box.
            tracks_data["x"] = tracks_data["x"] + tracks_data["length"] / 2
            tracks_data["y"] = tracks_data["y"] + tracks_data["width"] / 2
            tracks_data["heading"] = np.arctan2(tracks_data["vy"], tracks_data["vx"])
            # agent_id -> {scene_id}_{agent_id}
            tracks_data["agent_id"] = tracks_data["agent_id"].apply(
                lambda x: f"{scene_id_str}_{x}"
            )
            # "height" is unavailable in the HighD dataset
            index_cols = ["agent_id", "scene_ts"]
            tracks_data = tracks_data[
                ["heading"] + STATE_COLS + EXTENT_COLS[:-1] + index_cols
            ]
            tracks_data.set_index(["agent_id", "scene_ts"], inplace=True)
            tracks_data.sort_index(inplace=True)
            tracks_data.reset_index(level=1, inplace=True)
            scene_data = (
                pd.read_csv(
                    Path(self.metadata.data_dir) / f"{scene_id_str}_recordingMeta.csv"
                )
                .iloc[0]
                .to_dict()
            )
            self.dataset_obj[raw_data_idx] = {
                "scene_id": scene_id,
                "tracks_data": tracks_data,
                "scene_data": scene_data,
                "tracks_metadata": tracks_metadata,
            }

    def _get_location_from_scene_info(self, scene_info: Dict) -> str:
        return str(scene_info["scene_id"])

    def _get_matching_scenes_from_obj(
        self,
        scene_tag: SceneTag,
        scene_desc_contains: Optional[List[str]],
        env_cache: EnvCache,
    ) -> List[SceneMetadata]:
        all_scenes_list: List[HighDRecord] = list()
        scenes_list: List[SceneMetadata] = list()
        for raw_data_idx, scene_info in self.dataset_obj.items():
            scene_id = raw_data_idx + 1
            scene_location = self._get_location_from_scene_info(scene_info)
            scene_length: int = scene_info["tracks_data"]["scene_ts"].max().item() + 1
            all_scenes_list.append(
                HighDRecord(raw_data_idx, scene_length, scene_location)
            )
            scene_metadata = SceneMetadata(
                env_name=self.metadata.name,
                name=str(scene_id),
                dt=self.metadata.dt,
                raw_data_idx=raw_data_idx,
            )
            scenes_list.append(scene_metadata)
        self.cache_all_scenes_list(env_cache, all_scenes_list)
        return scenes_list

    def _get_matching_scenes_from_cache(
        self,
        scene_tag: SceneTag,
        scene_desc_contains: Optional[List[str]],
        env_cache: EnvCache,
    ) -> List[Scene]:
        all_scenes_list: List[HighDRecord] = env_cache.load_env_scenes_list(self.name)
        scenes_list: List[Scene] = list()
        for scene_record in all_scenes_list:
            data_idx, scene_length, scene_location = scene_record
            scene_id = data_idx + 1
            scene_metadata = Scene(
                self.metadata,
                str(scene_id),
                scene_location,
                HIGHD_SPLIT_NAME,
                scene_length,
                data_idx,
                None,
            )
            scenes_list.append(scene_metadata)
        return scenes_list

    def get_scene(self, scene_info: SceneMetadata) -> Scene:
        _, scene_name, _, data_idx = scene_info
        scene_data: pd.DataFrame = self.dataset_obj[data_idx]["tracks_data"]
        scene_location: str = self._get_location_from_scene_info(
            self.dataset_obj[data_idx]
        )
        scene_split: str = self.metadata.scene_split_map[scene_name]
        scene_length: int = scene_data["scene_ts"].max().item() + 1
        return Scene(
            self.metadata,
            scene_name,
            scene_location,
            scene_split,
            scene_length,
            data_idx,
            None,
        )

    def get_agent_info(
        self, scene: Scene, cache_path: Path, cache_class: Type[SceneCache]
    ) -> Tuple[List[AgentMetadata], List[List[AgentMetadata]]]:
        scene_data: pd.DataFrame = self.dataset_obj[scene.raw_data_idx][
            "tracks_data"
        ].copy()
        agent_list: List[AgentMetadata] = list()
        agent_presence: List[List[AgentMetadata]] = [
            [] for _ in range(scene.length_timesteps)
        ]
        for agent_id, frames in scene_data.groupby("agent_id")["scene_ts"]:
            start_frame: int = frames.iat[0].item()
            last_frame: int = frames.iat[-1].item()
            agent_metadata = self.dataset_obj[scene.raw_data_idx][
                "tracks_metadata"
            ].loc[agent_id.split("_")[1]]
            assert start_frame == agent_metadata["initialFrame"]
            assert last_frame == agent_metadata["finalFrame"]
            agent_info = AgentMetadata(
                name=str(agent_id),
                agent_type=AgentType.VEHICLE,
                first_timestep=start_frame,
                last_timestep=last_frame,
                extent=VariableExtent(),
            )
            agent_list.append(agent_info)
            for frame in frames:
                agent_presence[frame].append(agent_info)
        cache_class.save_agent_data(
            scene_data,
            cache_path,
            scene,
        )
        return agent_list, agent_presence

    def cache_map(
        self,
        data_idx: int,
        cache_path: Path,
        map_cache_class: Type[SceneCache],
        map_params: Dict[str, Any],
    ) -> None:
        env_name = self.metadata.name
        resolution = map_params["px_per_m"]
        raster_map = (
            cv2.imread(
                Path(self.metadata.data_dir)
                / f"{str(data_idx + 1).zfill(2)}_highway.png"
            ).astype(np.float32)
            / 255.0
        )
        raster_map = cv2.resize(
            raster_map,
            (
                math.ceil(HIGHD_PX_PER_M * resolution * raster_map.shape[1]),
                math.ceil(HIGHD_PX_PER_M * resolution * raster_map.shape[0]),
            ),
            interpolation=cv2.INTER_AREA,
        ).transpose(2, 0, 1)
        raster_from_world = np.eye(3)
        raster_from_world[:2, :2] *= resolution
        raster_metadata = RasterizedMapMetadata(
            name=f"{data_idx + 1}_map",
            shape=raster_map.shape,
            layers=["road", "lane", "shoulder"],
            layer_rgb_groups=([0], [1], [2]),
            resolution=map_params["px_per_m"],
            map_from_world=raster_from_world,
        )
        map_cache_class.cache_raster_map(
            env_name, str(data_idx), cache_path, raster_map, raster_metadata, map_params
        )

    def cache_maps(
        self,
        cache_path: Path,
        map_cache_class: Type[SceneCache],
        map_params: Dict[str, Any],
    ):
        for data_idx in range(HIGHD_NUM_SCENES):
            self.cache_map(data_idx, cache_path, map_cache_class, map_params)
