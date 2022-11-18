from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from trajdata.maps.map_kdtree import MapElementKDTree
    from trajdata.caching.scene_cache import SceneCache

from pathlib import Path
from typing import Dict

from trajdata.maps.vec_map import VectorMap
from trajdata.proto.vectorized_map_pb2 import VectorizedMap
from trajdata.utils import map_utils


class MapAPI:
    def __init__(self, unified_cache_path: Path) -> None:
        self.unified_cache_path: Path = unified_cache_path
        self.maps: Dict[str, VectorMap] = dict()

    def get_map(
        self, map_id: str, scene_cache: Optional[SceneCache] = None, **kwargs
    ) -> VectorMap:
        if map_id not in self.maps:
            env_name, map_name = map_id.split(":")
            env_maps_path: Path = self.unified_cache_path / env_name / "maps"
            stored_vec_map: VectorizedMap = map_utils.load_vector_map(
                env_maps_path / f"{map_name}.pb"
            )

            vec_map: VectorMap = VectorMap.from_proto(stored_vec_map, **kwargs)
            vec_map.search_kdtrees: Dict[
                str, MapElementKDTree
            ] = map_utils.load_kdtrees(env_maps_path / f"{map_name}_kdtrees.dill")

            self.maps[map_id] = vec_map

        if scene_cache is not None:
            self.maps[map_id].associate_scene_data(
                scene_cache.get_traffic_light_status_dict()
            )

        return self.maps[map_id]
