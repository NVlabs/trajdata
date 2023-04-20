import unittest
from pathlib import Path
from typing import Dict, List

from trajdata import MapAPI, VectorMap


class TestVectorMap(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cache_path = Path("~/.unified_data_cache").expanduser()
        cls.map_api = MapAPI(cache_path)
        cls.proto_loading_kwargs = {
            "incl_road_lanes": True,
            "incl_road_areas": True,
            "incl_ped_crosswalks": True,
            "incl_ped_walkways": True,
        }

        cls.location_dict: Dict[str, List[str]] = {
            "nuplan_mini": ["boston", "singapore", "pittsburgh", "las_vegas"],
            "nusc_mini": ["boston-seaport", "singapore-onenorth"],
            "lyft_sample": ["palo_alto"],
        }

    # TODO(pkarkus) this assumes we already have the maps cached. It would be better
    # to attempt to cache them if the cache does not yet exists.
    def test_map_existence(self):
        for env_name, map_names in self.location_dict.items():
            for map_name in map_names:
                vec_map: VectorMap = self.map_api.get_map(
                    f"{env_name}:{map_name}", **self.proto_loading_kwargs
                )
                assert vec_map is not None

    def test_proto_equivalence(self):
        for env_name, map_names in self.location_dict.items():
            for map_name in map_names:
                vec_map: VectorMap = self.map_api.get_map(
                    f"{env_name}:{map_name}", **self.proto_loading_kwargs
                )

                assert maps_equal(
                    VectorMap.from_proto(
                        vec_map.to_proto(), **self.proto_loading_kwargs
                    ),
                    vec_map,
                )

    # TODO(bivanovic): Add more!


def maps_equal(map1: VectorMap, map2: VectorMap) -> bool:
    elements1_set = set([elem.id for elem in map1.iter_elems()])
    elements2_set = set([elem.id for elem in map1.iter_elems()])
    return elements1_set == elements2_set


if __name__ == "__main__":
    unittest.main()
