from typing import NamedTuple


class EUPedsRecord(NamedTuple):
    name: str
    location: str
    length: str
    split: str


class NuscSceneRecord(NamedTuple):
    name: str
    location: str
    length: str
    desc: str


class LyftSceneRecord(NamedTuple):
    name: str
    length: str
