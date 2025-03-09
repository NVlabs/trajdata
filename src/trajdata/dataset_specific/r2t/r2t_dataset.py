import os
import os.path as osp
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union
import pandas as pd
from tqdm import tqdm
import numpy as np

from trajdata.caching import EnvCache, SceneCache
from trajdata.data_structures.agent import (
    Agent,
    AgentMetadata,
    AgentType,
    FixedExtent,
)
from trajdata.data_structures.environment import EnvMetadata
from trajdata.data_structures.scene_metadata import Scene, SceneMetadata
from trajdata.data_structures.scene_tag import SceneTag
from trajdata.dataset_specific.raw_dataset import RawDataset
from trajdata.dataset_specific.scene_records import R2tSceneRecord
from trajdata.utils import arr_utils

from .r2t_utils import R2T_DATASET_NAME, R2T_DT, R2T_SPLITS, NUSCENES_CLASS_MAPPING, nusc_type_to_unified_type


class Rank2TellTrajdataDataset(RawDataset):

    def compute_metadata(self, env_name: str, data_dir: str) -> EnvMetadata:
        if env_name != R2T_DATASET_NAME:
            raise ValueError(f"Unknown rank2tell env name: {env_name}")

        scene_id_to_split = {}
        for split in R2T_SPLITS:
            split_path = osp.join(data_dir, 'processed', f'{split}_split.txt')
            with open(split_path, 'r') as f:
                scenario_ids = [line.strip() for line in f]
            for scenario_id in scenario_ids:
                scene_id_to_split[scenario_id] = split

        return EnvMetadata(
            name=env_name,
            data_dir=data_dir,
            dt=R2T_DT,
            parts=[R2T_SPLITS],
            scene_split_map=scene_id_to_split,
            map_locations=None,
        )

    def _parse_3d_labels(self, label_dir, frame_idx):
        """
        Parse “labels_3d1_yyy.txt” into bboxes_3d
        Format (example):
          class_name, track_id, state, c_x, c_y, c_z, l_x, l_y, l_z, yaw
        """
        label_name = f"labels_3d1_{frame_idx:03d}.txt"
        label_file = osp.join(label_dir, label_name)

        bboxes_3d = []

        with open(label_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',')
                cls_name = parts[0].strip()
                label_id = NUSCENES_CLASS_MAPPING.get(cls_name, -1)
                if label_id == -1:
                    raise ValueError(f"Unknown class name: {cls_name}")

                # c_x, c_y, c_z, l_x, l_y, l_z, yaw
                instance_token = parts[1]
                # status = float(parts[2])  # e.g. static, moving
                c_x = float(parts[3])
                c_y = float(parts[4])
                c_z = float(parts[5])
                l_x = float(parts[6])
                l_y = float(parts[7])
                l_z = float(parts[8])
                yaw  = float(parts[9])

                # bbox_3d = [instance_token, frame_idx, label_id, c_x, c_y, c_z, l_x, l_y, l_z, yaw]
                bbox_3d = {'agent_id': instance_token, 'scene_ts': frame_idx, 'agent_type': label_id, 'x': c_x, 'y': c_y, 'z': c_z, 'length': l_x, 'width': l_y, 'height': l_z, 'heading': yaw}

                bboxes_3d.append(bbox_3d)

        return bboxes_3d

    def load_dataset_obj(self, verbose: bool = False) -> None:
        if verbose:
            print(f"Loading {self.name} dataset...", flush=True)

        if os.path.exists('dataset_obj.pkl'):
            self.dataset_obj = pd.read_pickle('dataset_obj.pkl')
            return

        self.dataset_obj = {}
        for scenario_id, split in tqdm(self.metadata.scene_split_map.items()):
            scenario_folder = osp.join(self.metadata.data_dir, 'scenarios', f'scenario_{scenario_id}')
            label_dir = osp.join(scenario_folder, 'labels')

            # Gather label files
            label_files = sorted(
                f for f in os.listdir(label_dir)
                if f.startswith('labels_3d1_') and f.endswith('.txt')
            )
            frame_indices_sorted = sorted(
                int(lf.replace('labels_3d1_', '').replace('.txt', ''))
                for lf in label_files
            )
            self.dataset_obj[scenario_id] = []
            for idx, frame_idx in enumerate(frame_indices_sorted):
                # only the ego info
                ego_data = self._parse_ego_odom(osp.join(scenario_folder, 'odom'), frame_idx)
                self.dataset_obj[scenario_id].append(ego_data)

                # other-agent info
                bboxes_3d = self._parse_3d_labels(label_dir, frame_idx)
                # transform each agent by ego position and rotation to convert from ego-centric to world-centric
                for bbox_3d in bboxes_3d:
                    # x, y, z
                    x = bbox_3d['x']
                    y = bbox_3d['y']
                    # yaw
                    yaw = bbox_3d['heading']
                    # ego position
                    ego_x = ego_data['x']
                    ego_y = ego_data['y']
                    ego_yaw = ego_data['heading']
                    # transform
                    x_w = x * np.cos(ego_yaw) - y * np.sin(ego_yaw) + ego_x
                    y_w = x * np.sin(ego_yaw) + y * np.cos(ego_yaw) + ego_y
                    yaw_w = yaw + ego_yaw
                    # update
                    bbox_3d['x'] = x_w
                    bbox_3d['y'] = y_w
                    bbox_3d['heading'] = yaw_w
                    self.dataset_obj[scenario_id].append(bbox_3d)

            # convert to dataframe
            self.dataset_obj[scenario_id] = pd.DataFrame(self.dataset_obj[scenario_id])
            # plot to check transform was done correctly
            # if True:
            #     import matplotlib.pyplot as plt
            #     df_interpolated = self.dataset_obj[scenario_id]
            #     # plot the agents. plot the interpolated regions in red
            #     translations_np = df_interpolated[['x', 'y']].to_numpy()
            #     yaws_np = df_interpolated['heading'].to_numpy()

            #     fig, ax = plt.subplots(figsize=(10, 10))
            #     # first plot agents
            #     for i, (agent_id, data) in enumerate(df_interpolated.groupby("agent_id")):
            #         ax.plot(data['x'], data['y'], 'b')
            #         # interp = interps[i]
            #         # for region in interp:
            #         #     ax.plot(translations_np[region, 0], translations_np[region, 1], 'r')
            #     plt.show()
            #     import ipdb; ipdb.set_trace()

        # save
        pd.to_pickle(self.dataset_obj, 'dataset_obj.pkl')

    def _parse_ego_odom(self, odom_dir, frame_idx):
        odom_file = osp.join(odom_dir, f'odom_{frame_idx:03d}.txt')
        l_x = 4.084
        l_y = 1.730
        l_z = 1.562
        with open(odom_file, 'r') as f_odom:
            line = f_odom.readline().strip()
            parts = line.split()
            x = float(parts[0])
            y = float(parts[1])
            z = float(parts[2])
            # roll = float(parts[3])
            # pitch = float(parts[4])
            yaw = float(parts[5])
        return {'agent_id': 'ego', 'scene_ts': frame_idx, 'agent_type': NUSCENES_CLASS_MAPPING['car'], 'x': x, 'y': y, 'z': z, 'length': l_x, 'width': l_y, 'height': l_z, 'heading': yaw}

    def _get_matching_scenes_from_obj(
        self,
        scene_tag: SceneTag,
        scene_desc_contains: Optional[List[str]],
        env_cache: EnvCache,
    ) -> List[SceneMetadata]:
        all_scenes_list: List[R2tSceneRecord] = list()

        scenes_list: List[SceneMetadata] = list()
        for idx, (scenario_id, scene_df) in enumerate(self.dataset_obj.items()):
            scene_name: str = scenario_id
            scene_length: int = scene_df["scene_ts"].max().item() + 1
            # scene_length: int = #scene_record['scene_ts'].unique().shape[0]

            # Saving all scene records for later caching.
            all_scenes_list.append(
                R2tSceneRecord(scene_name, scene_length, idx)
            )

            scene_split: str = self.metadata.scene_split_map[scene_name]
            if scene_split not in scene_tag:
                continue

            scene_metadata = SceneMetadata(
                env_name=self.metadata.name,
                name=scene_name,
                dt=self.metadata.dt,
                raw_data_idx=idx,
            )
            scenes_list.append(scene_metadata)

        self.cache_all_scenes_list(env_cache, all_scenes_list)
        return scenes_list

    def _get_matching_scenes_from_cache(
        self,
        scene_tag: SceneTag,
        scene_desc_contains: Optional[List[str]],
        env_cache: EnvCache,
    ) -> List[Scene]:
        all_scenes_list: List[R2tSceneRecord] = env_cache.load_env_scenes_list(self.name)

        scenes_list: List[SceneMetadata] = list()
        for scene_record in all_scenes_list:
            scene_name, scene_length, data_idx = scene_record
            scene_split: str = self.metadata.scene_split_map[scene_name]

            if scene_split not in scene_tag:
                continue

            scene_metadata = Scene(
                env_metadata=self.metadata,
                name=scene_name,
                location=scene_name,
                data_split=scene_split,
                length_timesteps=scene_length,  # AV2_SCENARIO_OBS_TIMESTEPS if data_split == "test" else AV2_SCENARIO_TOTAL_TIMESTEPS,
                raw_data_idx=data_idx,
                data_access_info=None,
            )
            scenes_list.append(scene_metadata)

        return scenes_list

    def get_scene(self, scene_info: SceneMetadata) -> Scene:
        _, scene_name, _, data_idx = scene_info

        scene_record = self.dataset_obj[scene_name]
        scene_split: str = self.metadata.scene_split_map[scene_name]
        scene_length: int = scene_record["scene_ts"].max().item() + 1

        return Scene(
            self.metadata,
            scene_name,
            scene_name,
            scene_split,
            scene_length,
            data_idx,
            scene_record,
            None,
        )
    
    def get_agent_info(
        self, scene: Scene, cache_path: Path, cache_class: Type[SceneCache]
    ) -> Tuple[List[AgentMetadata], List[List[AgentMetadata]]]:
        scene_data: pd.DataFrame = self.dataset_obj[scene.name].copy()

        scene_data.set_index(["agent_id", "scene_ts"], inplace=True)
        scene_data.sort_index(inplace=True)
        scene_data = scene_data[~scene_data.index.duplicated(keep='first')]  # Keeps first occurrence of duplicates s.t. each agent, ts pair is unique
        scene_data.reset_index(inplace=True)

        agent_ids: np.ndarray = scene_data.index.get_level_values(0).to_numpy()

        agent_list: List[AgentMetadata] = list()
        agent_presence: List[List[AgentMetadata]] = [
            [] for _ in range(scene.length_timesteps)
        ]
        agents_to_remove: List[int] = list()

        for agent_id, data in scene_data.groupby("agent_id"):
            frames = data["scene_ts"]
            if frames.shape[0] <= 1:
                # There are some agents with only a single detection to them, we don't care about these.
                agents_to_remove.append(agent_id)
                continue

            start_frame: int = frames.iat[0].item()
            last_frame: int = frames.iat[-1].item()

            if frames.shape[0] < last_frame - start_frame + 1:
                ## if missing frames, turn into two or more agents
                # get the contiguous frames
                frames = np.array(frames)
                contiguous_frames = np.split(frames, np.where(np.diff(frames) != 1)[0] + 1)
                for idx, contiguous_frame in enumerate(contiguous_frames):
                    if len(contiguous_frame) < 2:
                        continue
                    start_frame = contiguous_frame[0]
                    last_frame = contiguous_frame[-1]
                    assert last_frame<1000
                    new_agent_id = f'{agent_id}_{idx}'
                    agent_metadata = AgentMetadata(
                        name=new_agent_id,
                        agent_type=nusc_type_to_unified_type(data['agent_type'].iloc[0]),
                        first_timestep=start_frame,
                        last_timestep=last_frame,
                        extent=FixedExtent(length=data['length'].iloc[0], width=data['width'].iloc[0], height=data['height'].iloc[0]),
                    )
                    agent_list.append(agent_metadata)
                    for frame in contiguous_frame:
                        agent_presence[frame].append(agent_metadata)

                    # update df with new agent_ids
                    scene_data.loc[scene_data['scene_ts'].isin(contiguous_frame) & (scene_data['agent_id'] == agent_id), 'agent_id'] = new_agent_id
            else:
                assert last_frame<1000
                agent_metadata = AgentMetadata(
                    name=agent_id,
                    agent_type=nusc_type_to_unified_type(data['agent_type'].iloc[0]),
                    first_timestep=start_frame,
                    last_timestep=last_frame,
                    extent=FixedExtent(length=data['length'].iloc[0], width=data['width'].iloc[0], height=data['height'].iloc[0]),
                )

                agent_list.append(agent_metadata)
                for frame in frames:
                    agent_presence[frame].append(agent_metadata)

        # if True:
        #     import matplotlib.pyplot as plt
        #     translations_np = scene_data[['x', 'y']].to_numpy()
        #     fig, ax = plt.subplots(figsize=(10, 10))

        #     for i, (agent_id, data) in enumerate(scene_data.groupby("agent_id")):
        #         ax.plot(data['x'], data['y'])#, 'b')
        #         # interp = interps[i]
        #         # for region in interp:
        #             # ax.plot(translations_np[region, 0], translations_np[region, 1], 'r')

        #     plt.suptitle(scene.name)
        #     # plt.show()
        #     plt.savefig(f'../viz/r2t_visualization/r2t_interp_missing_frames/r2t_interpolated_{scene.name}_new_agent.png')
        #     plt.close()

        ### Calculating agent velocities
        scene_data.set_index(["agent_id"], inplace=True)
        scene_data[["vx", "vy"]] = (
            arr_utils.agent_aware_diff(scene_data[["x", "y"]].to_numpy(), agent_ids)
            / R2T_DT
        )

        ### Calculating agent accelerations
        scene_data[["ax", "ay"]] = (
            arr_utils.agent_aware_diff(scene_data[["vx", "vy"]].to_numpy(), agent_ids)
            / R2T_DT
        )

        # Removing agents with only one detection.
        scene_data.drop(index=agents_to_remove, inplace=True)

        # Changing the agent_id dtype to str
        scene_data.reset_index(inplace=True)
        scene_data["agent_id"] = scene_data["agent_id"].astype(str)
        scene_data.set_index(["agent_id", "scene_ts"], inplace=True)

        cache_class.save_agent_data(
            scene_data,
            cache_path,
            scene,
        )

        return agent_list, agent_presence

    def cache_map(
        self,
        map_name: str,
        layer_names: List[str],
        cache_path: Path,
        map_cache_class: Type[SceneCache],
        resolution: float,
    ) -> None:
        """
        No maps in this dataset!
        """
        pass

    def cache_maps(
        self,
        cache_path: Path,
        map_cache_class: Type[SceneCache],
        map_params: Dict[str, Any],
    ) -> None:
        """
        No maps in this dataset!
        """
        pass
