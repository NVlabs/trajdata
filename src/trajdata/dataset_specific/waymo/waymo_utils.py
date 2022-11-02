import pathlib
from typing import List

import numpy as np
import pandas as pd
import tensorflow as tf
import copy

from waymo_open_dataset.protos import scenario_pb2

from src.trajdata.utils import arr_utils

WAYMO_DT = 0.1

data_dir = "../../../../../scenarios"
from trajdata.data_structures.agent import (
    Agent,
    AgentMetadata,
    AgentType,
    FixedExtent, VariableExtent
)


def load_single_tfrecord(file_name, verbose=False):
    dataset = tf.data.TFRecordDataset(file_name, compression_type='')
    dataset_it = dataset.as_numpy_iterator()
    scenarios = []
    for data in dataset_it:
        scenario = scenario_pb2.Scenario()
        scenario.ParseFromString(bytearray(data))
        scenarios.append(copy.deepcopy(scenario))

    if verbose:
        print("\n\nFile name: " + file_name)
        print("Num scenes: ", len(scenarios))
        print("Scenario 1 info:")
        print("\tID: ", scenarios[0].scenario_id)
        print("\tTimestamps: ", scenarios[0].timestamps_seconds)
        print("\tCurrent Timestamp: ", scenarios[0].current_time_index)
        print("\tNum Tracks: ", len(scenarios[0].tracks))
        print("\tNum Dynamic States: ", len(scenarios[0].dynamic_map_states))
        print("\tNum Map Features: ", len(scenarios[0].map_features))
        print("\tsdc_track_index ", scenarios[0].sdc_track_index)
        tracks = scenarios[516].tracks[scenarios[516].sdc_track_index - 2]
        for i in range(91):
            print(tracks.states[i].length, tracks.states[i].width, tracks.states[i].height)
            print(tracks.states[i].valid)

    return scenarios


def load_tfrecords(source_dir, verbose=False):
    scenarios = []
    source_it = pathlib.Path().glob(source_dir + "/*.tfrecord")
    file_names = [str(file_name) for file_name in source_it if file_name.is_file()]
    for file_name in file_names:
        scenarios.extend(load_single_tfrecord(file_name, verbose))
    return scenarios


# print("Training Data: ")
# load_tfrecords(data_dir + '/training', True)
# print("Training_20s Data: ")
# load_tfrecords(data_dir + '/training_20s', True)
# print("Testing Data: ")
# load_tfrecords(data_dir + '/testing', True)
# print("Testing_interactive Data: ")
# load_tfrecords(data_dir + '/testing_interactive', True)
# print("Validation Data: ")
# load_tfrecords(data_dir + '/validation', True)
# print("Validation_interactive Data: ")
# load_tfrecords(data_dir + '/validation_interactive', True)


def translate_agent_type(type):
    if type == scenario_pb2.Track.ObjectType.TYPE_VEHICLE:
        return AgentType.VEHICLE
    if type == scenario_pb2.Track.ObjectType.TYPE_PEDESTRIAN:
        return AgentType.PEDESTRIAN
    if type == scenario_pb2.Track.ObjectType.TYPE_CYCLIST:
        return AgentType.BICYCLE
    if type == scenario_pb2.Track.ObjectType.OTHER:
        return AgentType.UNKNOWN
    return -1


class WaymoScenarios:
    def __init__(self, source_dir):
        self.name = source_dir
        self.scenarios = load_tfrecords(source_dir)


agent_list: List[AgentMetadata] = []
agent_presence: List[List[AgentMetadata]] = [
    [] for _ in range(91)
]
scenario = load_tfrecords(data_dir + '/training', False)[0]
agent_ids = []
agent_translations = []
agent_velocities = []
agent_yaws = []
agent_ml_class = []
agent_sizes = []

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
    for timestep in range(91):
        if states[timestep].valid:
            first_timestep = timestep
            break
    last_timestep = 90
    for timestep in range(91):
        if states[90 - timestep].valid:
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

agent_ids = np.repeat(agent_ids, 91)

agent_translations = np.array(agent_translations)
agent_velocities = np.array(agent_velocities)
agent_sizes = np.array(agent_sizes)

agent_ml_class = np.repeat(agent_ml_class, 91)
agent_yaws = np.array(agent_yaws)

print(agent_ids.shape)
print(agent_translations.shape)
print(agent_velocities.shape)
print(agent_sizes.shape)
print(agent_ml_class.shape)
print(agent_yaws.shape)

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
    np.arange(91), 63*91
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
        / WAYMO_DT
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
all_agent_data_df.reset_index(inplace=True)
all_agent_data_df["agent_id"] = all_agent_data_df["agent_id"].astype(str)
all_agent_data_df.set_index(["agent_id", "scene_ts"], inplace=True)

print(all_agent_data_df)
print(all_agent_data_df.columns)
print(all_agent_data_df.loc[:, final_cols])
print(pd.concat([all_agent_data_df.loc[:, final_cols]]))
# print(scenario.tracks[0].id)
# print(scenario.tracks[0].states[1].height)

# for track in scenario.tracks:
#
#     print(all_agent_data_df['height'][str(track.id)][0])
#     break
