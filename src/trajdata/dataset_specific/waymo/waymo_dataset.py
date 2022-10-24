from collections import defaultdict
from functools import partial

from trajdata.caching import EnvCache, SceneCache
from trajdata.dataset_specific.raw_dataset import RawDataset
from waymo_utils import WaymoScenarios
import waymo_utils
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from ..scene_records import WaymoSceneRecord
from trajdata.data_structures.scene_tag import SceneTag
from trajdata.data_structures import (
    AgentMetadata,
    EnvMetadata,
    Scene,
    SceneMetadata,
    SceneTag,
)
from waymo_open_dataset.protos.scenario_pb2 import Scenario
def const_lambda(const_val: Any) -> Any:
    return const_val

class WaymoDataset(RawDataset):
    def compute_metadata(self, env_name: str, data_dir: str) -> EnvMetadata:
        if env_name == "waymo_train":

            # nuScenes possibilities are the Cartesian product of these
            dataset_parts = [
                ("train"),
                ("san francisco", "mountain view", "los angeles", "detroit", "seattle", "phoenix")
            ]
            scene_split_map = defaultdict(partial(const_lambda, const_val="train"))
        elif env_name == "waymo_val":


            # nuScenes possibilities are the Cartesian product of these
            dataset_parts = [
                ("val"),
                ("san francisco", "mountain view", "los angeles", "detroit", "seattle", "phoenix"),
            ]
            scene_split_map = defaultdict(partial(const_lambda, const_val="val"))
        elif env_name == "waymo_test":

            # nuScenes possibilities are the Cartesian product of these
            dataset_parts= [
                ("test"),
                ("san francisco", "mountain view", "los angeles", "detroit", "seattle", "phoenix"),
            ]
            scene_split_map = defaultdict(partial(const_lambda, const_val="test"))
        return EnvMetadata(name=env_name,
                           data_dir=data_dir,
                           dt=waymo_utils.WAYMO_DT,
                           parts=dataset_parts,
                           scene_split_map=scene_split_map
                           )

    def load_dataset_obj(self, verbose: bool = False) -> None:
        if verbose:
            print(f"Loading {self.name} dataset...", flush=True)

        self.dataset_obj = WaymoScenarios(self.metadata.data_dir)


    def _get_matching_scenes_from_obj(
            self,
            scene_tag: SceneTag,
            scene_desc_contains: Optional[List[str]],
            env_cache: EnvCache,
    ) -> List[SceneMetadata]:
        all_scenes_list: List[WaymoSceneRecord] = list()

        scenes_list: List[SceneMetadata] = list()
        for idx, scene_record in enumerate(self.dataset_obj.scenarios):
            scene_name: str = scene_record.scenario_id
            scene_split: str = self.metadata.scene_split_map[scene_name]
            scene_length: int = len(scene_record.timestamps)

            # Saving all scene records for later caching.
            all_scenes_list.append(WaymoSceneRecord(scene_name, str(scene_length), idx))

            if (scene_split in scene_tag and scene_desc_contains is None
            ):
                scene_metadata = SceneMetadata(
                    env_name=self.metadata.name,
                    name=scene_name,
                    dt=self.metadata.dt,
                    raw_data_idx=idx,
                )
                scenes_list.append(scene_metadata)

        self.cache_all_scenes_list(env_cache, all_scenes_list)
        return scenes_list

    def get_scene(self, scene_info: SceneMetadata) -> Scene:
        _, name, _, data_idx = scene_info
        scene_frames: Scenario = self.dataset_obj.scenarios[
            data_idx
        ]
        scene_name: str = name
        scene_split: str = self.metadata.scene_split_map[scene_name]
        scene_length: int = len(Scenario.timestamps_seconds)  # Doing .item() otherwise it'll be a numpy.int64

        return Scene(
            self.metadata,
            scene_name,
            '',
            scene_split,
            scene_length,
            data_idx,
            scene_frames,
        )

    def _get_matching_scenes_from_cache(
            self,
            scene_tag: SceneTag,
            scene_desc_contains: Optional[List[str]],
            env_cache: EnvCache,
    ) -> List[Scene]:
        all_scenes_list: List[WaymoSceneRecord] = env_cache.load_env_scenes_list(
            self.name
        )

        scenes_list: List[SceneMetadata] = list()
        for scene_record in all_scenes_list:
            scene_name, scene_length, data_idx = scene_record
            scene_split: str = self.metadata.scene_split_map[scene_name]

            if (
                    scene_split in scene_tag
                    and scene_desc_contains is None
            ):
                scene_metadata = Scene(
                    self.metadata,
                    scene_name,
                    "",
                    scene_split,
                    scene_length,
                    data_idx,
                    None,  # This isn't used if everything is already cached.
                )
                scenes_list.append(scene_metadata)

        return scenes_list