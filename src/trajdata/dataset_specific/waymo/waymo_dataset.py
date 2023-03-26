import warnings
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type
import numpy as np
import pandas as pd
import tqdm
import tensorflow as tf
from trajdata.dataset_specific.waymo import waymo_utils
from waymo_open_dataset.protos.scenario_pb2 import Scenario
from trajdata.dataset_specific.waymo.waymo_utils import WaymoScenarios, translate_agent_type, interpolate_array

from trajdata.caching import EnvCache, SceneCache
from trajdata.data_structures import (
    AgentMetadata,
    EnvMetadata,
    Scene,
    SceneMetadata,
    SceneTag,
)
from trajdata.data_structures.agent import (
    Agent,
    AgentMetadata,
    AgentType,
    FixedExtent,
    VariableExtent,
)
from trajdata.data_structures.scene_tag import SceneTag
from trajdata.dataset_specific.raw_dataset import RawDataset
from trajdata.maps import VectorMap
from trajdata.proto.vectorized_map_pb2 import (
    MapElement,
    PedCrosswalk,
    RoadLane,
    VectorizedMap,
)

from trajdata.utils import arr_utils
from trajdata.dataset_specific.scene_records import WaymoSceneRecord


def const_lambda(const_val: Any) -> Any:
    return const_val

def get_mode_val(series: pd.Series) -> float:
    return series.mode().iloc[0].item()


class WaymoDataset(RawDataset):
    def compute_metadata(self, env_name: str, data_dir: str) -> EnvMetadata:
        if env_name == "waymo_train":

            # Waymo possibilities are the Cartesian product of these
            dataset_parts = [
                ("train",)
            ]
            scene_split_map = defaultdict(partial(const_lambda, const_val="train"))
        elif env_name == "waymo_val":

            # Waymo possibilities are the Cartesian product of these
            dataset_parts = [
                ("val",)
            ]
            scene_split_map = defaultdict(partial(const_lambda, const_val="val"))
        elif env_name == "waymo_test":

            # Waymo possibilities are the Cartesian product of these
            dataset_parts = [
                ("test",)
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
        dataset_name: str = ""
        if self.name == "waymo_train":
            dataset_name = "training"
        elif self.name == "waymo_val":
            dataset_name = "validation"
        elif self.name == "waymo_test":
            dataset_name = "testing"
        self.dataset_obj = WaymoScenarios(dataset_name=dataset_name, source_dir=self.metadata.data_dir)

    def _get_matching_scenes_from_obj(
            self,
            scene_tag: SceneTag,
            scene_desc_contains: Optional[List[str]],
            env_cache: EnvCache,
    ) -> List[SceneMetadata]:
        all_scenes_list: List[WaymoSceneRecord] = list()

        scenes_list: List[SceneMetadata] = list()
        for idx in range(self.dataset_obj.num_scenarios):
            scene_name: str = "scene_"+str(idx)
            scene_split: str = self.metadata.scene_split_map[scene_name]
            scene_length: int = self.dataset_obj.scene_length

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
        scene_name: str = name
        scene_split: str = self.metadata.scene_split_map[scene_name]
        scene_length: int = self.dataset_obj.scene_length

        return Scene(
            self.metadata,
            scene_name,
            'unknown',
            scene_split,
            scene_length,
            data_idx,
            None
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
                    "unknown",
                    scene_split,
                    scene_length,
                    data_idx,
                    None,  # This isn't used if everything is already cached.
                )
                scenes_list.append(scene_metadata)

        return scenes_list

    def get_agent_info(
            self, scene: Scene, cache_path: Path, cache_class: Type[SceneCache]
    ) -> Tuple[List[AgentMetadata], List[List[AgentMetadata]]]:
        agent_list: List[AgentMetadata] = []
        agent_presence: List[List[AgentMetadata]] = [
            [] for _ in range(scene.length_timesteps)
        ]
        dataset = tf.data.TFRecordDataset([self.dataset_obj.get_filename(scene.raw_data_idx)], compression_type='')
        scenario: Scenario = Scenario()
        for data in dataset:
            scenario.ParseFromString(bytearray(data.numpy()))
            break
        agent_ids = []
        agent_translations = []
        agent_velocities = []
        agent_yaws = []
        agent_ml_class = []
        agent_sizes = []

        agents_to_remove = []
        for index, track in enumerate(scenario.tracks):
            agent_name = track.id
            if index == scenario.sdc_track_index:
                agent_name = "ego"
            agent_ids.append(agent_name)

            agent_type: AgentType = translate_agent_type(track.object_type)
            if agent_type == -1:
                continue
            agent_ml_class.append(agent_type)
            states = track.states
            translations = []
            velocities = []
            sizes = []
            yaws = []
            first_timestep = 0
            last_timestep = scene.length_timesteps - 1
            for state in states:
                if state.valid:
                    translations.append((state.center_x, state.center_y, state.center_z))
                    velocities.append((state.velocity_x, state.velocity_y))
                    sizes.append((state.length, state.width, state.height))
                    yaws.append(state.heading)
                else:
                    translations.append((0, 0, 0))
                    velocities.append((0, 0))
                    sizes.append((0, 0, 0))
                    yaws.append(0)
            agent_translations.extend(translations)
            agent_velocities.extend(velocities)
            agent_sizes.extend(sizes)
            agent_yaws.extend(yaws)


            # for timestep in range(scene.length_timesteps):
            #     if states[timestep].valid:
            #         first_timestep = timestep
            #         break
            # for timestep in range(scene.length_timesteps):
            #     if states[scene.length_timesteps - timestep - 1].valid:
            #         last_timestep = timestep
            #         break

            agent_info = AgentMetadata(
                name=agent_name,
                agent_type=agent_type,
                first_timestep=first_timestep,
                last_timestep=last_timestep,
                extent=VariableExtent(),
            )
            if last_timestep - first_timestep != 0:
                agent_list.append(agent_info)
                for timestep in range(first_timestep, last_timestep + 1):
                    agent_presence[timestep].append(agent_info)
            else:
                agents_to_remove.append(agent_name)

        agent_ids = np.repeat(agent_ids, scene.length_timesteps)
        agent_ml_class = np.repeat(agent_ml_class, scene.length_timesteps)

        agent_translations = np.array(agent_translations)
        agent_velocities = np.array(agent_velocities)
        agent_sizes = np.array(agent_sizes)
        agent_yaws = np.array(agent_yaws)
        for i in range(agent_translations.shape[1]):
            agent_translations[:, i] = interpolate_array(agent_translations[:, i])
        for i in range(agent_velocities.shape[1]):
            agent_velocities[:, i] = interpolate_array(agent_velocities[:, i])
        for i in range(agent_sizes.shape[1]):
            agent_sizes[:, i] = interpolate_array(agent_sizes[:, i])
        for i in range(agent_yaws.shape[1]):
            agent_yaws[:, i] = interpolate_array(agent_yaws[:, i])



        all_agent_data = np.concatenate(
            [
                agent_translations,
                agent_velocities,
                np.expand_dims(agent_yaws, axis=1),
                np.expand_dims(agent_ml_class, axis=1),
                agent_sizes,
            ],
            axis=1,
        )

        traj_cols = ["x", "y", "z", "vx", "vy", "heading"]
        class_cols = ["class_id"]
        extent_cols = ["length", "width", "height"]
        agent_frame_ids = np.resize(
            np.arange(scene.length_timesteps), len(scenario.tracks) * scene.length_timesteps
        )

        all_agent_data_df = pd.DataFrame(
            all_agent_data,
            columns=traj_cols + class_cols + extent_cols,
            index=[agent_ids, agent_frame_ids],
        )

        all_agent_data_df.index.names = ["agent_id", "scene_ts"]
        all_agent_data_df.sort_index(inplace=True)
        all_agent_data_df.reset_index(level=1, inplace=True)

        all_agent_data_df[["ax", "ay"]] = (
                arr_utils.agent_aware_diff(
                    all_agent_data_df[["vx", "vy"]].to_numpy(), agent_ids
                )
                / waymo_utils.WAYMO_DT
        )
        final_cols = [
                         "x",
                         "y",
                         "vx",
                         "vy",
                         "ax",
                         "ay",
                         "heading",
                     ] + extent_cols
        # Removing agents with only one detection.
        all_agent_data_df.drop(index=agents_to_remove, inplace=True)

        # Changing the agent_id dtype to str
        all_agent_data_df.reset_index(inplace=True)
        all_agent_data_df["agent_id"] = all_agent_data_df["agent_id"].astype(str)
        all_agent_data_df.set_index(["agent_id", "scene_ts"], inplace=True)

        cache_class.save_agent_data(
            all_agent_data_df.loc[:, final_cols],
            cache_path,
            scene,
        )

        return agent_list, agent_presence

    def cache_map(self,
                  data_idx: int,
                  cache_path: Path,
                  map_cache_class: Type[SceneCache],
                  map_params: Dict[str, Any]):
        dataset = tf.data.TFRecordDataset([self.dataset_obj.get_filename(data_idx)], compression_type='')
        scenario: Scenario = Scenario()
        for data in dataset:
            scenario.ParseFromString(bytearray(data.numpy()))
            break
        vector_map_proto: VectorizedMap = waymo_utils.extract_vectorized(
            map_features=scenario.map_features,
            map_name=f"{self.name}:{data_idx}")
        vector_map: VectorMap = VectorMap.from_proto(vector_map_proto, incl_road_lanes=True, incl_ped_crosswalks=True)

        vector_map.associate_scene_data(
            waymo_utils.extract_traffic_lights(
                dynamic_states=scenario.dynamic_map_states))
        map_cache_class.finalize_and_cache_map(cache_path, vector_map, map_params)

    def cache_maps(
            self,
            cache_path: Path,
            map_cache_class: Type[SceneCache],
            map_params: Dict[str, Any],
    ) -> None:
        for i in tqdm.trange(3):
            self.cache_map(i, cache_path, map_cache_class, map_params)
