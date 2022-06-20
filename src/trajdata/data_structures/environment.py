import itertools
from pathlib import Path
from typing import Dict, List, Tuple

from trajdata.data_structures.scene_tag import SceneTag


class EnvMetadata:
    def __init__(
        self,
        name: str,
        data_dir: str,
        dt: float,
        parts: List[Tuple[str]],
        scene_split_map: Dict[str, str],
    ) -> None:
        self.name = name
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.dt = dt
        self.parts = parts
        self.scene_tags: List[SceneTag] = [
            SceneTag(tag_tuple)
            # Cartesian product of the given list of tuples
            for tag_tuple in itertools.product(*([(name,)] + parts))
        ]
        self.scene_split_map = scene_split_map
