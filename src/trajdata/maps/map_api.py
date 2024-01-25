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
    def __init__(self, unified_cache_path: Path, keep_in_memory: bool = False) -> None:
        """A simple interface for loading trajdata's vector maps which does not require
        instantiation of a `UnifiedDataset` object.

        Args:
            unified_cache_path (Path): Path to trajdata's local cache on disk.
            keep_in_memory (bool): Whether loaded maps should be stored
            in memory (memoized) for later re-use. For most cases (e.g., batched dataloading),
            this is a good idea. However, this can cause rapid memory usage growth for some
            datasets (e.g., Waymo) and it can be better to disable this. Defaults to False.
        """
        self.unified_cache_path: Path = unified_cache_path
        self.maps: Dict[str, VectorMap] = dict()
        self._keep_in_memory = keep_in_memory

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
            vec_map.search_kdtrees = map_utils.load_kdtrees(
                env_maps_path / f"{map_name}_kdtrees.dill"
            )
            vec_map.search_rtrees = map_utils.load_rtrees(
                env_maps_path / f"{map_name}_rtrees.dill"
            )

            if self._keep_in_memory:
                self.maps[map_id] = vec_map
        else:
            vec_map = self.maps[map_id]

        if scene_cache is not None:
            vec_map.associate_scene_data(
                scene_cache.get_traffic_light_status_dict(
                    kwargs.get("desired_dt", None)
                )
            )

        return vec_map
