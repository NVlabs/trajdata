from typing import List

from trajdata.data_structures.scene_tag import SceneTag


def pretty_string_tags(tag_lst: List[SceneTag]) -> List[str]:
    return [str(tag) for tag in tag_lst]
