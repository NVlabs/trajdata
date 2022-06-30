from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type

import numpy as np
from torch.utils.data import Dataset

from trajdata.caching import EnvCache, SceneCache
from trajdata.data_structures import Scene, SceneMetadata
from trajdata.parallel.temp_cache import TemporaryCache
from trajdata.utils import agent_utils, scene_utils
from trajdata.utils.env_utils import get_raw_dataset


def scene_paths_collate_fn(filled_scenes: List) -> List:
    return filled_scenes


class ParallelDatasetPreprocessor(Dataset):
    def __init__(
        self,
        scene_info_list: List[SceneMetadata],
        envs_dir_dict: Dict[str, str],
        env_cache_path: str,
        temp_cache_path: str,
        desired_dt: Optional[float],
        cache_class: Type[SceneCache],
        rebuild_cache: bool,
    ) -> None:
        self.env_cache_path = np.array(env_cache_path).astype(np.string_)
        self.temp_cache_path = np.array(temp_cache_path).astype(np.string_)
        self.desired_dt = desired_dt
        self.cache_class = cache_class
        self.rebuild_cache = rebuild_cache

        env_names: List[str] = list(envs_dir_dict.keys())
        scene_names: List[str] = [scene_info.name for scene_info in scene_info_list]

        self.scene_idxs = np.array(
            [scene_info.raw_data_idx for scene_info in scene_info_list], dtype=int
        )
        self.env_name_idxs = np.array(
            [env_names.index(scene_info.env_name) for scene_info in scene_info_list],
            dtype=int,
        )
        self.scene_name_idxs = np.array(
            [scene_names.index(scene_info.name) for scene_info in scene_info_list],
            dtype=int,
        )

        self.env_names_arr = np.array(env_names).astype(np.string_)
        self.scene_names_arr = np.array(scene_names).astype(np.string_)
        self.data_dir_arr = np.array(list(envs_dir_dict.values())).astype(np.string_)

        self.data_len: int = len(scene_info_list)

    def __len__(self) -> int:
        return self.data_len

    def __getitem__(self, idx: int) -> Tuple[Path, Path]:
        env_cache_path: Path = Path(str(self.env_cache_path, encoding="utf-8"))
        env_cache: EnvCache = EnvCache(env_cache_path)

        env_idx: int = self.env_name_idxs[idx]
        scene_idx: int = self.scene_name_idxs[idx]

        env_name: str = str(self.env_names_arr[env_idx], encoding="utf-8")
        raw_dataset = get_raw_dataset(
            env_name, str(self.data_dir_arr[env_idx], encoding="utf-8")
        )

        scene_name: str = str(self.scene_names_arr[scene_idx], encoding="utf-8")

        scene_info = SceneMetadata(
            env_name, scene_name, raw_dataset.metadata.dt, self.scene_idxs[idx]
        )

        # Leaving verbose False here so that we don't spam
        # stdout with loading messages.
        raw_dataset.load_dataset_obj(verbose=False)
        scene: Scene = agent_utils.get_agent_data(
            scene_info, raw_dataset, env_cache, self.rebuild_cache, self.cache_class
        )
        raw_dataset.del_dataset_obj()

        orig_scene_path: Path = EnvCache.scene_metadata_path(
            env_cache.path, scene.env_name, scene.name
        )
        if scene_utils.enforce_desired_dt(scene, self.desired_dt):
            temp_cache_path: str = str(self.temp_cache_path, encoding="utf-8")
            return (
                str(orig_scene_path),
                str(TemporaryCache(temp_cache_path).cache(scene)),
            )
        else:
            return (str(orig_scene_path), None)
