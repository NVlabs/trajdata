from enum import IntEnum


class TrafficLightStatus(IntEnum):
    NO_DATA = -1
    UNKNOWN = 0
    GREEN = 1
    RED = 2
