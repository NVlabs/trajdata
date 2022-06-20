from typing import Type

from trajdata.caching import EnvCache, SceneCache
from trajdata.data_structures import Scene, SceneMetadata
from trajdata.dataset_specific import RawDataset


def get_agent_data(
    scene_info: SceneMetadata,
    raw_dataset: RawDataset,
    env_cache: EnvCache,
    rebuild_cache: bool,
    cache_class: Type[SceneCache],
) -> Scene:
    if not rebuild_cache and env_cache.scene_is_cached(
        scene_info.env_name, scene_info.name
    ):
        return env_cache.load_scene(scene_info.env_name, scene_info.name)

    scene: Scene = raw_dataset.get_scene(scene_info)
    agent_list, agent_presence = raw_dataset.get_agent_info(
        scene, env_cache.path, cache_class
    )

    scene.update_agent_info(agent_list, agent_presence)
    env_cache.save_scene(scene)

    return scene
