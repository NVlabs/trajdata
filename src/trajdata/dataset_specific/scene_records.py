from typing import NamedTuple


class Argoverse2Record(NamedTuple):
    name: str
    data_idx: int


class EUPedsRecord(NamedTuple):
    name: str
    location: str
    length: str
    split: str
    data_idx: int


class SDDPedsRecord(NamedTuple):
    name: str
    length: str
    data_idx: int


class InteractionRecord(NamedTuple):
    name: str
    length: str
    data_idx: int


class NuscSceneRecord(NamedTuple):
    name: str
    location: str
    length: str
    desc: str
    data_idx: int


class VODSceneRecord(NamedTuple):
    token: str
    name: str
    location: str
    length: str
    desc: str
    data_idx: int


class LyftSceneRecord(NamedTuple):
    name: str
    length: str
    data_idx: int


class WaymoSceneRecord(NamedTuple):
    name: str
    length: str
    data_idx: int


class NuPlanSceneRecord(NamedTuple):
    name: str
    location: str
    length: str
    split: str
    # desc: str
    data_idx: int
