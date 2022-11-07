from collections import defaultdict
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from trajdata.data_structures.agent import (
    Agent,
    AgentMetadata,
    AgentType,
    FixedExtent,
    VariableExtent
)
from trajdata.caching import EnvCache, SceneCache
from trajdata.dataset_specific.raw_dataset import RawDataset
from waymo_utils import WaymoScenarios, translate_agent_type
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

from ...utils import arr_utils


def const_lambda(const_val: Any) -> Any:
    return const_val

def get_mode_val(series: pd.Series) -> float:
    return series.mode().iloc[0].item()
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
            dataset_parts = [
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

    def get_agent_info(
            self, scene: Scene, cache_path: Path, cache_class: Type[SceneCache]
    ) -> Tuple[List[AgentMetadata], List[List[AgentMetadata]]]:
        agent_list: List[AgentMetadata] = []
        agent_presence: List[List[AgentMetadata]] = [
            [] for _ in range(scene.length_timesteps)
        ]
        scenario = self.dataset_obj.scenarios[scene.raw_data_idx]
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
            agent_ml_class.append(agent_type)
            states = track.states
            translations = [[state.center_x, state.center_y, state.center_z] for state in states]
            agent_translations.extend(translations)
            velocities = [[state.velocity_x, state.velocity_y] for state in states]
            agent_velocities.extend(velocities)
            sizes = [[state.length, state.width, state.height] for state in states]
            agent_sizes.extend(sizes)
            yaws = [state.heading for state in states]
            agent_yaws.extend(yaws)

            first_timestep = 0
            states = track.states
            for timestep in range(scene.length_timesteps):
                if states[timestep].valid:
                    first_timestep = timestep
                    break
            last_timestep = scene.length_timesteps - 1
            for timestep in range(scene.length_timesteps):
                if states[scene.length_timesteps - timestep - 1].valid:
                    last_timestep = timestep
                    break

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

        agent_translations = np.array(agent_translations)
        agent_velocities = np.array(agent_velocities)
        agent_sizes = np.array(agent_sizes)

        agent_ml_class = np.repeat(agent_ml_class, scene.length_timesteps)
        agent_yaws = np.array(agent_yaws)
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
            pd.concat([all_agent_data_df.loc[:, final_cols]]),
            cache_path,
            scene,
        )

        return agent_list, agent_presence
