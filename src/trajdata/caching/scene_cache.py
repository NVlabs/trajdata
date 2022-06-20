from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd

from trajdata.augmentation.augmentation import Augmentation
from trajdata.data_structures.agent import AgentMetadata
from trajdata.data_structures.map import Map, MapMetadata
from trajdata.data_structures.scene_metadata import Scene


class SceneCache:
    def __init__(
        self,
        cache_path: Path,
        scene: Scene,
        scene_ts: int,
        augmentations: Optional[List[Augmentation]] = None,
    ) -> None:
        """
        Creates and prepares the cache for online data loading.
        """
        self.path = cache_path
        self.scene = scene
        self.dt = scene.dt
        self.scene_ts = scene_ts
        self.augmentations = augmentations

        # Ensuring the scene cache folder exists
        self.scene_dir: Path = self.path / self.scene.env_name / self.scene.name
        self.scene_dir.mkdir(parents=True, exist_ok=True)

    # AGENT STATE DATA
    @staticmethod
    def save_agent_data(
        agent_data: pd.DataFrame,
        cache_path: Path,
        scene: Scene,
    ) -> None:
        raise NotImplementedError()

    def get_value(self, agent_id: str, scene_ts: int, attribute: str) -> float:
        """
        Get a single attribute value for an agent at a timestep.
        """
        raise NotImplementedError()

    def get_state(self, agent_id: str, scene_ts: int) -> np.ndarray:
        """
        Get an agent's state at a specific timestep.
        """
        raise NotImplementedError()

    def transform_data(self, **kwargs) -> None:
        """
        Transform the data before accessing it later, e.g., to make the mean zero or rotate the scene around an agent.
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

    def get_positions_at(
        self, scene_ts: int, agents: List[AgentMetadata]
    ) -> np.ndarray:
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

    # MAPS
    @staticmethod
    def are_maps_cached(cache_path: Path, env_name: str) -> bool:
        raise NotImplementedError()

    @staticmethod
    def is_map_cached(cache_path: Path, env_name: str, map_name: str) -> bool:
        raise NotImplementedError()

    @staticmethod
    def cache_map(cache_path: Path, map_obj: Map, env_name: str) -> None:
        raise NotImplementedError()

    @staticmethod
    def cache_map_layers(
        cache_path: Path,
        map_info: MapMetadata,
        layer_fn: Callable[[str], np.ndarray],
        env_name: str,
    ) -> None:
        raise NotImplementedError()

    def load_map_patch(
        self,
        world_x: float,
        world_y: float,
        desired_patch_size: int,
        resolution: int,
        offset_xy: Tuple[float, float],
        agent_heading: float,
        return_rgb: bool,
        rot_pad_factor: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError()
