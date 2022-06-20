from pathlib import Path
from typing import List, NamedTuple, Union

import dill

from trajdata.data_structures.scene_metadata import Scene


class EnvCache:
    def __init__(self, cache_location: Path) -> None:
        self.path = cache_location

    def env_is_cached(self, env_name: str) -> bool:
        return (self.path / env_name / "scenes_list.dill").is_file()

    def scene_is_cached(self, env_name: str, scene_name: str) -> bool:
        return EnvCache.scene_metadata_path(self.path, env_name, scene_name).is_file()

    @staticmethod
    def scene_metadata_path(base_path: Path, env_name: str, scene_name: str) -> Path:
        return base_path / env_name / scene_name / "scene_metadata.dill"

    def load_scene(self, env_name: str, scene_name: str) -> Scene:
        scene_file: Path = EnvCache.scene_metadata_path(self.path, env_name, scene_name)
        with open(scene_file, "rb") as f:
            scene: Scene = dill.load(f)

        return scene

    def save_scene(self, scene: Scene) -> None:
        scene_file: Path = EnvCache.scene_metadata_path(
            self.path, scene.env_name, scene.name
        )

        scene_cache_dir: Path = scene_file.parent
        scene_cache_dir.mkdir(parents=True, exist_ok=True)

        with open(scene_file, "wb") as f:
            dill.dump(scene, f)

    def load_env_scenes_list(self, env_name: str) -> List[NamedTuple]:
        env_cache_dir: Path = self.path / env_name
        with open(env_cache_dir / "scenes_list.dill", "rb") as f:
            scenes_list: List[NamedTuple] = dill.load(f)

        return scenes_list

    def save_env_scenes_list(
        self, env_name: str, scenes_list: List[NamedTuple]
    ) -> None:
        env_cache_dir: Path = self.path / env_name
        env_cache_dir.mkdir(parents=True, exist_ok=True)
        with open(env_cache_dir / "scenes_list.dill", "wb") as f:
            dill.dump(scenes_list, f)

    @staticmethod
    def load(scene_info_path: Union[Path, str]) -> Scene:
        with open(scene_info_path, "rb") as handle:
            scene: Scene = dill.load(handle)

        return scene
