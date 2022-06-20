from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Optional, Union

import dill

from trajdata.data_structures.scene_metadata import Scene


class TemporaryCache:
    def __init__(self, temp_dir: Optional[str] = None) -> None:
        self.temp_dir: Optional[TemporaryDirectory] = None
        if temp_dir is None:
            self.temp_dir: TemporaryDirectory = TemporaryDirectory()
            self.path: Path = Path(self.temp_dir.name)
        else:
            self.path: Path = Path(temp_dir)

    def cache(self, scene: Scene, ret_str: bool = False) -> Union[Path, str]:
        tmp_file_path: Path = self.path / TemporaryCache.get_file_path(scene)
        with open(tmp_file_path, "wb") as f:
            dill.dump(scene, f)

        if ret_str:
            return str(tmp_file_path)
        else:
            return tmp_file_path

    def cache_scenes(self, scenes: List[Scene]) -> List[str]:
        paths: List[str] = list()
        for scene in scenes:
            tmp_file_path: Path = self.path / TemporaryCache.get_file_path(scene)
            with open(tmp_file_path, "wb") as f:
                dill.dump(scene, f)

            paths.append(str(tmp_file_path))

        return paths

    def cleanup(self) -> None:
        self.temp_dir.cleanup()

    @staticmethod
    def get_file_path(scene_info: Scene) -> Path:
        return f"{scene_info.env_name}_{scene_info.name}.dill"
