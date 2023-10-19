from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import pandas as pd
import tqdm

from ysdc_dataset_api.utils import get_file_paths, scenes_generator
from ysdc_dataset_api.proto import Scene as YSDCScene
from trajdata.caching import EnvCache, SceneCache
from trajdata.data_structures import EnvMetadata, Scene, SceneMetadata, SceneTag
from trajdata.data_structures.agent import AgentMetadata
from trajdata.dataset_specific.raw_dataset import RawDataset
from trajdata.dataset_specific.scene_records import YandexShiftsSceneRecord
from trajdata.dataset_specific.yandex_shifts import yandex_shifts_utils
from trajdata.maps import VectorMap
from trajdata.utils.parallel_utils import parallel_apply
from trajdata.dataset_specific.yandex_shifts.yandex_shifts_utils import (
    read_scene_from_original_proto,
    get_scene_path,
    extract_vectorized,
    extract_traffic_light_status,
    extract_agent_data_from_ysdc_scene,
)


def const_lambda(const_val: Any) -> Any:
    return const_val


class YandexShiftsDataset(RawDataset):
    def compute_metadata(self, env_name: str, data_dir: str) -> EnvMetadata:
        if env_name == "ysdc_train":
            dataset_parts = [("train",)]
            scene_split_map = defaultdict(partial(const_lambda, const_val="train"))

        elif env_name == "ysdc_development":
            dataset_parts = [("development",)]
            scene_split_map = defaultdict(
                partial(const_lambda, const_val="development")
            )

        elif env_name == "ysdc_eval":
            dataset_parts = [("eval",)]
            scene_split_map = defaultdict(partial(const_lambda, const_val="eval"))

        elif env_name == "ysdc_full":
            dataset_parts = [("full",)]
            scene_split_map = defaultdict(partial(const_lambda, const_val="full"))

        return EnvMetadata(
            name=env_name,
            data_dir=data_dir,
            dt=yandex_shifts_utils.YSDC_DT,
            parts=dataset_parts,
            scene_split_map=scene_split_map,
        )

    def load_dataset_obj(self, verbose: bool = False) -> None:
        if verbose:
            print(f"Loading {self.name} dataset...", flush=True)
        self.dataset_obj = scenes_generator(get_file_paths(self.metadata.data_dir))

    def _get_matching_scenes_from_obj(
        self,
        scene_tag: SceneTag,
        scene_desc_contains: Optional[List[str]],
        env_cache: EnvCache,
    ) -> List[SceneMetadata]:
        all_scenes_list: List[YandexShiftsSceneRecord] = list()
        scenes_list: List[SceneMetadata] = list()
        for idx, scene in tqdm.tqdm(
            enumerate(self.dataset_obj), desc="Processing scenes from proto files"
        ):
            scene_name: str = scene.id
            scene_split: str = self.metadata.scene_split_map[scene_name]
            scene_length: int = yandex_shifts_utils.YSDC_LENGTH
            # Saving all scene records for later caching.
            all_scenes_list.append(
                YandexShiftsSceneRecord(
                    scene_name,
                    str(scene_length),
                    idx,
                    scene.scene_tags.day_time,
                    scene.scene_tags.season,
                    scene.scene_tags.track,
                    scene.scene_tags.sun_phase,
                    scene.scene_tags.precipitation,
                )
            )
            if scene_split in scene_tag and scene_desc_contains is None:
                scene_metadata = SceneMetadata(
                    env_name=self.metadata.name,
                    name=scene_name,
                    dt=self.metadata.dt,
                    raw_data_idx=idx,
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
        all_scenes_list: List[YandexShiftsSceneRecord] = env_cache.load_env_scenes_list(
            self.name
        )
        scenes_list: List[SceneMetadata] = list()
        for scene_record in all_scenes_list:
            scene_split: str = self.metadata.scene_split_map[scene_record.name]
            if scene_split in scene_tag and scene_desc_contains is None:
                scene_metadata = Scene(
                    self.metadata,
                    scene_record.name,
                    scene_record.data_idx,
                    scene_split,
                    scene_record.length,
                    scene_record.data_idx,
                    None,  # This isn't used if everything is already cached.
                )
                scenes_list.append(scene_metadata)
        return scenes_list

    def get_scene(self, scene_info: SceneMetadata) -> Scene:
        _, _, _, data_idx = scene_info
        scene_data_from_proto: YSDCScene = read_scene_from_original_proto(
            get_scene_path(self.metadata.data_dir, data_idx)
        )
        num_history_timestamps = len(scene_data_from_proto.past_vehicle_tracks)
        num_future_timestamps = len(scene_data_from_proto.future_vehicle_tracks)
        scene_name: str = scene_data_from_proto.id
        scene_split: str = self.metadata.scene_split_map[scene_name]
        scene_length: int = num_history_timestamps + num_future_timestamps
        return Scene(
            self.metadata,
            scene_data_from_proto.id,
            data_idx,
            scene_split,
            scene_length,
            data_idx,
            {
                "day_time": scene_data_from_proto.scene_tags.day_time,
                "season": scene_data_from_proto.scene_tags.season,
                "track_location": scene_data_from_proto.scene_tags.track,
                "sun_phase": scene_data_from_proto.scene_tags.sun_phase,
                "precipitation": scene_data_from_proto.scene_tags.precipitation,
            },
        )

    def get_agent_info(
        self, scene: Scene, cache_path: Path, cache_class: Type[SceneCache]
    ) -> Tuple[List[AgentMetadata], List[List[AgentMetadata]]]:
        scene_data_from_proto = read_scene_from_original_proto(
            get_scene_path(self.metadata.data_dir, scene.raw_data_idx)
        )
        (
            scene_agents_data_df,
            agent_list,
            agent_presence,
        ) = extract_agent_data_from_ysdc_scene(scene_data_from_proto, scene)
        cache_class.save_agent_data(scene_agents_data_df, cache_path, scene)
        tls_dict = extract_traffic_light_status(scene_data_from_proto)
        tls_df = pd.DataFrame(
            tls_dict.values(),
            index=pd.MultiIndex.from_tuples(
                tls_dict.keys(), names=["lane_id", "scene_ts"]
            ),
            columns=["status"],
        )
        cache_class.save_traffic_light_data(tls_df, cache_path, scene)
        return agent_list, agent_presence

    def cache_map(
        self,
        data_idx: int,
        cache_path: Path,
        map_cache_class: Type[SceneCache],
        map_params: Dict[str, Any],
    ):
        scene_data_from_proto = read_scene_from_original_proto(
            get_scene_path(self.metadata.data_dir, data_idx)
        )
        vector_map: VectorMap = extract_vectorized(
            scene_data_from_proto.path_graph, map_name=f"{self.name}:{data_idx}"
        )
        map_cache_class.finalize_and_cache_map(cache_path, vector_map, map_params)

    def cache_maps(
        self,
        cache_path: Path,
        map_cache_class: Type[SceneCache],
        map_params: Dict[str, Any],
    ) -> None:
        num_workers: int = map_params.get("num_workers", 0)
        if num_workers > 1:
            parallel_apply(
                partial(
                    self.cache_map,
                    cache_path=cache_path,
                    map_cache_class=map_cache_class,
                    map_params=map_params,
                ),
                range(len(get_file_paths(self.metadata.data_dir))),
                num_workers=num_workers,
            )

        else:
            for i in tqdm.trange(len(get_file_paths(self.metadata.data_dir))):
                self.cache_map(i, cache_path, map_cache_class, map_params)
