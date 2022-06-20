from typing import Any, Dict, Final, List, Union

import numpy as np
import pandas as pd
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

from trajdata.data_structures import Agent, AgentMetadata, AgentType, FixedExtent, Scene
from trajdata.utils import arr_utils

NUSC_DT: Final[float] = 0.5


def frame_iterator(nusc_obj: NuScenes, scene: Scene) -> Dict[str, Union[str, int]]:
    """Loops through all frames in a scene and yields them for the caller to deal with the information."""
    curr_scene_token: str = scene.data_access_info["first_sample_token"]
    while curr_scene_token:
        frame = nusc_obj.get("sample", curr_scene_token)

        yield frame

        curr_scene_token = frame["next"]


def agent_iterator(nusc_obj: NuScenes, frame_info: Dict[str, Any]) -> Dict[str, Any]:
    """Loops through all annotations (agents) in a frame and yields them for the caller to deal with the information."""
    ann_token: str
    for ann_token in frame_info["anns"]:
        ann_record = nusc_obj.get("sample_annotation", ann_token)

        agent_category: str = ann_record["category_name"]
        if agent_category.startswith("vehicle") or agent_category.startswith("human"):
            yield ann_record


def get_ego_pose(nusc_obj: NuScenes, frame_info: Dict[str, Any]) -> Dict[str, Any]:
    cam_front_data = nusc_obj.get("sample_data", frame_info["data"]["CAM_FRONT"])
    ego_pose = nusc_obj.get("ego_pose", cam_front_data["ego_pose_token"])
    return ego_pose


def agg_agent_data(
    nusc_obj: NuScenes,
    agent_data: Dict[str, Any],
    curr_scene_index: int,
    frame_idx_dict: Dict[str, int],
) -> Agent:
    """Loops through all annotations of a specific agent in a scene and aggregates their data into an Agent object."""
    if agent_data["prev"]:
        print("WARN: This is not the first frame of this agent!")

    translation_list = [np.array(agent_data["translation"][:2])[np.newaxis]]
    agent_size = agent_data["size"]
    yaw_list = [Quaternion(agent_data["rotation"]).yaw_pitch_roll[0]]

    prev_idx: int = curr_scene_index
    curr_sample_ann_token: str = agent_data["next"]
    while curr_sample_ann_token:
        agent_data = nusc_obj.get("sample_annotation", curr_sample_ann_token)

        translation = np.array(agent_data["translation"][:2])
        heading = Quaternion(agent_data["rotation"]).yaw_pitch_roll[0]
        curr_idx: int = frame_idx_dict[agent_data["sample_token"]]
        if curr_idx > prev_idx + 1:
            fill_time = np.arange(prev_idx + 1, curr_idx)
            xs = np.interp(
                x=fill_time,
                xp=[prev_idx, curr_idx],
                fp=[translation_list[-1][0, 0], translation[0]],
            )
            ys = np.interp(
                x=fill_time,
                xp=[prev_idx, curr_idx],
                fp=[translation_list[-1][0, 1], translation[1]],
            )
            headings: np.ndarray = arr_utils.angle_wrap(
                np.interp(
                    x=fill_time,
                    xp=[prev_idx, curr_idx],
                    fp=np.unwrap([yaw_list[-1], heading]),
                )
            )
            translation_list.append(np.stack([xs, ys], axis=1))
            yaw_list.extend(headings.tolist())

        translation_list.append(translation[np.newaxis])
        # size_list.append(agent_data['size'])
        yaw_list.append(heading)

        prev_idx = curr_idx
        curr_sample_ann_token = agent_data["next"]

    translations_np = np.concatenate(translation_list, axis=0)

    # Doing this prepending so that the first velocity isn't zero (rather it's just the first actual velocity duplicated)
    prepend_pos = translations_np[0] - (translations_np[1] - translations_np[0])
    velocities_np = (
        np.diff(translations_np, axis=0, prepend=np.expand_dims(prepend_pos, axis=0))
        / NUSC_DT
    )

    # Doing this prepending so that the first acceleration isn't zero (rather it's just the first actual acceleration duplicated)
    prepend_vel = velocities_np[0] - (velocities_np[1] - velocities_np[0])
    accelerations_np = (
        np.diff(velocities_np, axis=0, prepend=np.expand_dims(prepend_vel, axis=0))
        / NUSC_DT
    )

    anno_yaws_np = np.expand_dims(np.stack(yaw_list, axis=0), axis=1)
    # yaws_np = np.expand_dims(
    #     np.arctan2(velocities_np[:, 1], velocities_np[:, 0]), axis=1
    # )
    # sizes_np = np.stack(size_list, axis=0)

    # import matplotlib.pyplot as plt

    # fig, ax = plt.subplots()
    # ax.plot(translations_np[:, 0], translations_np[:, 1], color="blue")
    # ax.quiver(
    #     translations_np[:, 0],
    #     translations_np[:, 1],
    #     np.cos(anno_yaws_np),
    #     np.sin(anno_yaws_np),
    #     color="green",
    #     label="annotated heading"
    # )
    # ax.quiver(
    #     translations_np[:, 0],
    #     translations_np[:, 1],
    #     np.cos(yaws_np),
    #     np.sin(yaws_np),
    #     color="orange",
    #     label="velocity heading"
    # )
    # ax.scatter([translations_np[0, 0]], [translations_np[0, 1]], color="red", label="Start", zorder=20)
    # ax.legend(loc='best')
    # plt.show()

    agent_data_np = np.concatenate(
        [translations_np, velocities_np, accelerations_np, anno_yaws_np], axis=1
    )
    last_timestep = curr_scene_index + agent_data_np.shape[0] - 1
    agent_data_df = pd.DataFrame(
        agent_data_np,
        columns=["x", "y", "vx", "vy", "ax", "ay", "heading"],
        index=pd.MultiIndex.from_tuples(
            [
                (agent_data["instance_token"], idx)
                for idx in range(curr_scene_index, last_timestep + 1)
            ],
            names=["agent_id", "scene_ts"],
        ),
    )

    agent_type = nusc_type_to_unified_type(agent_data["category_name"])
    agent_metadata = AgentMetadata(
        name=agent_data["instance_token"],
        agent_type=agent_type,
        first_timestep=curr_scene_index,
        last_timestep=last_timestep,
        extent=FixedExtent(
            length=agent_size[1], width=agent_size[0], height=agent_size[2]
        ),
    )
    return Agent(
        metadata=agent_metadata,
        data=agent_data_df,
    )


def nusc_type_to_unified_type(nusc_type: str) -> AgentType:
    if nusc_type.startswith("human"):
        return AgentType.PEDESTRIAN
    elif nusc_type == "vehicle.bicycle":
        return AgentType.BICYCLE
    elif nusc_type == "vehicle.motorcycle":
        return AgentType.MOTORCYCLE
    elif nusc_type.startswith("vehicle"):
        return AgentType.VEHICLE
    else:
        return AgentType.UNKNOWN


def agg_ego_data(nusc_obj: NuScenes, scene: Scene) -> Agent:
    translation_list: List[np.ndarray] = list()
    yaw_list: List[float] = list()
    for frame_info in frame_iterator(nusc_obj, scene):
        ego_pose = get_ego_pose(nusc_obj, frame_info)
        yaw_list.append(Quaternion(ego_pose["rotation"]).yaw_pitch_roll[0])
        translation_list.append(ego_pose["translation"][:2])

    translations_np: np.ndarray = np.stack(translation_list, axis=0)

    # Doing this prepending so that the first velocity isn't zero (rather it's just the first actual velocity duplicated)
    prepend_pos: np.ndarray = translations_np[0] - (
        translations_np[1] - translations_np[0]
    )
    velocities_np: np.ndarray = (
        np.diff(translations_np, axis=0, prepend=np.expand_dims(prepend_pos, axis=0))
        / NUSC_DT
    )

    # Doing this prepending so that the first acceleration isn't zero (rather it's just the first actual acceleration duplicated)
    prepend_vel: np.ndarray = velocities_np[0] - (velocities_np[1] - velocities_np[0])
    accelerations_np: np.ndarray = (
        np.diff(velocities_np, axis=0, prepend=np.expand_dims(prepend_vel, axis=0))
        / NUSC_DT
    )

    yaws_np: np.ndarray = np.expand_dims(np.stack(yaw_list, axis=0), axis=1)
    # yaws_np = np.expand_dims(np.arctan2(velocities_np[:, 1], velocities_np[:, 0]), axis=1)

    ego_data_np: np.ndarray = np.concatenate(
        [translations_np, velocities_np, accelerations_np, yaws_np], axis=1
    )
    ego_data_df = pd.DataFrame(
        ego_data_np,
        columns=["x", "y", "vx", "vy", "ax", "ay", "heading"],
        index=pd.MultiIndex.from_tuples(
            [("ego", idx) for idx in range(ego_data_np.shape[0])],
            names=["agent_id", "scene_ts"],
        ),
    )

    ego_metadata = AgentMetadata(
        name="ego",
        agent_type=AgentType.VEHICLE,
        first_timestep=0,
        last_timestep=ego_data_np.shape[0] - 1,
        extent=FixedExtent(length=4.084, width=1.730, height=1.562),
    )
    return Agent(
        metadata=ego_metadata,
        data=ego_data_df,
    )
