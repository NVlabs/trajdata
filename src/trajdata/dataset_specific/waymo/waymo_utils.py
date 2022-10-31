import pathlib

import tensorflow as tf
import copy

from waymo_open_dataset.protos import scenario_pb2

WAYMO_DT = 0.1

data_dir = "../../../../../scenarios"
from trajdata.data_structures.agent import (
    Agent,
    AgentMetadata,
    AgentType,
    FixedExtent
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
        tracks = scenarios[516].tracks[scenarios[516].sdc_track_index-2]
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


print("Training Data: ")
load_tfrecords(data_dir + '/training', True)
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
