from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trajdata.maps import TrafficLightStatus, VectorMap


from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from trajdata.augmentation.augmentation import Augmentation
from trajdata.data_structures.agent import AgentMetadata
from trajdata.data_structures.scene_metadata import Scene


class SceneCache:
    def __init__(
        self,
        cache_path: Path,
        scene: Scene,
        augmentations: Optional[List[Augmentation]] = None,
    ) -> None:
        """
        Creates and prepares the cache for online data loading.
        """
        self.path = cache_path
        self.scene = scene
        self.dt = scene.dt
        self.augmentations = augmentations

        # Ensuring the scene cache folder exists
        self.scene_dir: Path = SceneCache.scene_cache_dir(
            self.path, self.scene.env_name, self.scene.name
        )
        self.scene_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def scene_cache_dir(cache_path: Path, env_name: str, scene_name: str) -> Path:
        """Standardized convention to compute scene cache folder path"""
        return cache_path / env_name / scene_name

    def write_cache_to_disk(self) -> None:
        """Saves agent data to disk for fast loading later (just like save_agent_data),
        but using the class attributes for the sources of data and file paths.
        """
        raise NotImplementedError()

    # AGENT STATE DATA
    @staticmethod
    def save_agent_data(
        agent_data: Any,
        cache_path: Path,
        scene: Scene,
    ) -> None:
        """Saves agent data to disk for fast loading later."""
        raise NotImplementedError()

    def get_value(self, agent_id: str, scene_ts: int, attribute: str) -> float:
        """
        Get a single attribute value for an agent at a timestep.
        """
        raise NotImplementedError()

    def get_raw_state(self, agent_id: str, scene_ts: int) -> np.ndarray:
        """
        Get an agent's raw state (without transformations applied)
        """
        raise NotImplementedError()

    def get_state(self, agent_id: str, scene_ts: int) -> np.ndarray:
        """
        Get an agent's state at a specific timestep.
        """
        raise NotImplementedError()

    def get_states(self, agent_ids: List[str], scene_ts: int) -> np.ndarray:
        """
        Get multiple agents' states at a specific timestep.
        """
        raise NotImplementedError()

    def transform_data(self, **kwargs) -> None:
        """
        Transform the data before accessing it later, e.g., to make the mean zero or rotate the scene around an agent.
        This can either be done in this function call or just stored for later lazy application.
        """
        raise NotImplementedError()

    def reset_transforms(self) -> None:
        """
        Transform the data back to its original coordinate system.
        This can either be done in this function call or just stored for later lazy application.
        """
        raise NotImplementedError()

    def interpolate_data(self, desired_dt: float, method: str = "linear") -> None:
        """Increase the sampling frequency of the data by interpolation.

        Args:
            desired_dt (float): The desired spacing between timesteps.
            method (str, optional): The type of interpolation to use, currently only "linear" is implemented. Defaults to "linear".
        """
        raise NotImplementedError()

    def get_agent_history(
        self,
        agent_info: AgentMetadata,
        scene_ts: int,
        history_sec: Tuple[Optional[float], Optional[float]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError()

    def get_agent_future(
        self,
        agent_info: AgentMetadata,
        scene_ts: int,
        future_sec: Tuple[Optional[float], Optional[float]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError()

    def get_agents_history(
        self,
        scene_ts: int,
        agents: List[AgentMetadata],
        history_sec: Tuple[Optional[float], Optional[float]],
    ) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
        raise NotImplementedError()

    def get_agents_future(
        self,
        scene_ts: int,
        agents: List[AgentMetadata],
        future_sec: Tuple[Optional[float], Optional[float]],
    ) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
        raise NotImplementedError()

    # TRAFFIC LIGHT INFO
    @staticmethod
    def save_traffic_light_data(
        traffic_light_status_data: Any, cache_path: Path, scene: Scene
    ) -> None:
        """Saves traffic light status to disk for easy access later"""
        raise NotImplementedError()

    def is_traffic_light_data_cached(self, desired_dt: Optional[float] = None) -> bool:
        raise NotImplementedError()

    def get_traffic_light_status_dict(
        self,
    ) -> Dict[Tuple[int, int], TrafficLightStatus]:
        """Returns lookup table for traffic light status in the current scene
        lane_id, scene_ts -> TrafficLightStatus"""
        raise NotImplementedError()

    # MAPS
    @staticmethod
    def are_maps_cached(cache_path: Path, env_name: str) -> bool:
        raise NotImplementedError()

    @staticmethod
    def is_map_cached(
        cache_path: Path, env_name: str, map_name: str, resolution: float
    ) -> bool:
        raise NotImplementedError()

    @staticmethod
    def finalize_and_cache_map(
        cache_path: Path,
        vector_map: VectorMap,
        map_params: Dict[str, Any],
    ) -> None:
        raise NotImplementedError()

    def load_map_patch(
        self,
        world_x: float,
        world_y: float,
        desired_patch_size: int,
        resolution: float,
        offset_xy: Tuple[float, float],
        agent_heading: float,
        return_rgb: bool,
        rot_pad_factor: float = 1.0,
        no_map_val: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        raise NotImplementedError()
