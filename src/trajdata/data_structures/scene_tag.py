from typing import Set, Tuple


class SceneTag:
    def __init__(self, tag_tuple: Tuple[str, ...]) -> None:
        self._tag_tuple: Set[str] = set(tag_tuple)

    def contains(self, query: Set[str]) -> bool:
        return query.issubset(self._tag_tuple)

    def __contains__(self, item) -> bool:
        return item in self._tag_tuple

    def __repr__(self) -> str:
        return "-".join(self._tag_tuple)
