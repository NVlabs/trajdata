# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

"""
USDZ Dataset implementation for trajdata.

This module implements trajdata's RawDataset interface for USDZ files by
wrapping the existing Artifact class in alpasim, which already provides stable USDZ parsing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, NamedTuple

import numpy as np
import pandas as pd

from trajdata.data_structures.environment import EnvMetadata
from trajdata.data_structures.scene_metadata import Scene
from trajdata.dataset_specific.raw_dataset import RawDataset
from trajdata.maps import VectorMap
from trajdata.utils import arr_utils

# Import Artifact for USDZ parsing
from alpasim_utils.artifact import Artifact

logger = logging.getLogger(__name__)


# Define UsdzSceneRecord locally (compatible with trajdata version if it exists)
class UsdzSceneRecord(NamedTuple):
    """Scene record for USDZ files."""
    name: str  # trajdata expects 'name', not 'scene_id'
    env_name: str
    data_dir: str
    dt: float
    raw_data_idx: int  # Required by trajdata
    usdz_path: str
    length_timesteps: int
    duration_s: float
    metadata: dict


class UsdzDataset(RawDataset):
    """
    Trajdata RawDataset implementation for USDZ files.

    This class wraps the existing Artifact class to leverage its stable
    USDZ parsing logic, converting data to trajdata's standard format.
    """

    def __init__(
        self,
        env_name: str,
        data_dir: str,
        parallelizable: bool = True,
        has_maps: bool = True,
        **kwargs
    ) -> None:
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

        self._smooth_trajectories = kwargs.get('smooth_trajectories', True)
        # Artifact instances keyed by scene_id
        self._artifacts: Dict[str, Artifact] = {}
        self._scene_records: List[UsdzSceneRecord] = []

        super().__init__(env_name, data_dir, parallelizable, has_maps)

    @staticmethod
    def compute_metadata(env_name: str, data_dir: str) -> EnvMetadata:
        """Compute dataset metadata."""
        return EnvMetadata(
            name=env_name,
            data_dir=data_dir,
            dt=0.1,
            parts=[],
            scene_split_map={},
        )

    def load_dataset_obj(self, verbose: bool = False) -> None:
        """
        Scan USDZ files using Artifact.discover_from_glob().

        This leverages Artifact's existing scanning logic.
        """
        logger.info(f"Scanning USDZ files in {self.data_dir}")

        # Use Artifact's discovery mechanism
        glob_pattern = str(self.data_dir / "**/*.usdz")
        try:
            artifacts_dict = Artifact.discover_from_glob(
                glob_pattern,
                recursive=True,
                smooth_trajectories=self._smooth_trajectories
            )
            logger.info(f"Found {len(artifacts_dict)} USDZ scenes")
        except Exception as e:
            logger.error(f"Failed to discover USDZ files: {e}")
            artifacts_dict = {}

        # Create scene records from artifacts
        for scene_id, artifact in artifacts_dict.items():
            try:
                # Extract info from artifact
                rig = artifact.rig

                # Get number of frames safely
                if rig.trajectory.timestamps_us is not None:
                    timestamps = rig.trajectory.timestamps_us
                    num_frames = len(timestamps)

                    if num_frames > 1:
                        duration_s = float(timestamps[-1] - timestamps[0]) / 1e6
                        dt = duration_s / (num_frames - 1)
                    else:
                        duration_s = 0.0
                        dt = 0.1
                else:
                    num_frames = 0
                    duration_s = 0.0
                    dt = 0.1

                record = UsdzSceneRecord(
                    name=scene_id,  # trajdata expects 'name'
                    env_name=self.name,
                    data_dir=str(self.data_dir),
                    dt=dt,
                    raw_data_idx=len(self._scene_records),  # Index in scene list
                    usdz_path=artifact.source,
                    length_timesteps=num_frames,
                    duration_s=duration_s,
                    metadata={'scene_id': scene_id, 'usdz_path': artifact.source},
                )

                self._scene_records.append(record)
                self._artifacts[scene_id] = artifact

                if verbose:
                    logger.debug(f"Loaded scene: {scene_id}")

            except Exception as e:
                logger.error(f"Failed to process {scene_id}: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                continue

        logger.info(f"Loaded {len(self._scene_records)} scenes from USDZ files")
        self.dataset_obj = {'artifacts': self._artifacts, 'scene_records': self._scene_records}

    def _get_matching_scenes_from_obj(
        self,
        scene_tag: str,
        scene_desc_contains: Optional[List[str]],
        env_cache: Any,
    ) -> List[UsdzSceneRecord]:
        """Return list of scene records matching the criteria."""
        # For USDZ, we simply return all scene records
        # (no filtering by tag/desc since USDZ doesn't have these)
        return self._scene_records

    def get_scene(self, scene_info: Any) -> Scene:
        """
        Load scene data using Artifact and convert to trajdata format.

        Args:
            scene_info: Scene metadata (UsdzSceneRecord with scene_id attribute)

        Returns:
            Scene object with trajectory and map data
        """
        # Extract scene_id from scene_info (could be str, or NamedTuple with .name)
        if isinstance(scene_info, str):
            scene_id = scene_info
        elif hasattr(scene_info, 'name'):
            scene_id = scene_info.name
        else:
            raise ValueError(f"Cannot extract scene_id from scene_info: {type(scene_info)}")

        if scene_id not in self._artifacts:
            raise KeyError(f"Scene {scene_id} not found in dataset (available: {list(self._artifacts.keys())})")

        artifact = self._artifacts[scene_id]
        logger.info(f"Loading scene {scene_id} from {Path(artifact.source).name}")

        # Get data from Artifact (already parsed!)
        rig = artifact.rig
        traffic_objects = artifact.traffic_objects
        vector_map = artifact.map if self.has_maps else None

        # Convert to trajdata DataFrame format
        agent_data = self._convert_to_trajdata_format(rig, traffic_objects)

        # Calculate scene length
        if rig.trajectory.timestamps_us is not None:
            length_timesteps = len(rig.trajectory.timestamps_us)
        else:
            length_timesteps = 0

        # Create Scene object with proper constructor parameters
        scene = Scene(
            env_metadata=self.metadata,
            name=scene_id,
            location="",
            data_split="",
            length_timesteps=length_timesteps,
            raw_data_idx=0,
            data_access_info={
                'usdz_path': artifact.source,
                'usdz_stem': Path(artifact.source).stem,
                'scene_id': scene_id,
            }
        )

        # Attach data for trajdata to cache
        scene.agent_data = agent_data
        scene.map_data = vector_map

        return scene

    def _convert_to_trajdata_format(
        self,
        rig: Any,
        traffic_objects: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Convert Artifact's Rig and TrafficObjects to trajdata DataFrame format.

        Args:
            rig: Rig object from Artifact
            traffic_objects: TrafficObjects dict from Artifact

        Returns:
            DataFrame with columns: agent_id, timestep, x, y, z, vx, vy, heading, etc.
        """
        # Calculate actual dt from timestamps
        if rig.trajectory.timestamps_us is not None and len(rig.trajectory.timestamps_us) > 1:
            timestamps_s = np.array(rig.trajectory.timestamps_us) / 1e6
            dt = float(np.mean(np.diff(timestamps_s)))
        else:
            dt = 0.1  # Fallback to 10Hz

        rows = []

        # Process ego vehicle
        traj = rig.trajectory
        if not traj.is_empty():
            try:
                num_poses = len(traj)
                for t_idx in range(num_poses):
                    pose = traj.get_pose(t_idx)

                    # Extract position (vec3 is numpy array)
                    pos = pose.vec3
                    x, y, z = float(pos[0]), float(pos[1]), float(pos[2])

                    # Extract heading from quaternion
                    # Artifact uses [x, y, z, w] order, convert to [w, x, y, z] for arr_utils
                    quat = pose.quat  # [x, y, z, w]
                    quat_wxyz = np.array([quat[3], quat[0], quat[1], quat[2]])  # [w, x, y, z]
                    heading = float(arr_utils.quaternion_to_yaw(quat_wxyz))

                    # Velocity: Trajectory class doesn't have velocities_mps
                    # Velocities will be calculated by trajdata from positions
                    vx, vy = 0.0, 0.0

                    rows.append({
                        'agent_id': 'ego',
                        'timestep': t_idx,
                        'x': x,
                        'y': y,
                        'z': z,
                        'vx': vx,
                        'vy': vy,
                        'heading': heading,
                        'length': float(rig.vehicle_config.aabb_x_m) if rig.vehicle_config else 4.5,
                        'width': float(rig.vehicle_config.aabb_y_m) if rig.vehicle_config else 2.0,
                        'agent_type': 'vehicle',
                    })
            except Exception as e:
                logger.warning(f"Failed to process ego trajectory: {e}")
                import traceback
                logger.debug(traceback.format_exc())

        # Process traffic objects
        for track_id, traffic_obj in traffic_objects.items():
            if traffic_obj.trajectory.is_empty():
                continue

            try:
                num_poses = len(traffic_obj.trajectory)
                for t_idx in range(num_poses):
                    pose = traffic_obj.trajectory.get_pose(t_idx)

                    # vec3 and quat are numpy arrays
                    pos = pose.vec3
                    x, y, z = float(pos[0]), float(pos[1]), float(pos[2])

                    # Convert quaternion [x, y, z, w] to [w, x, y, z] for arr_utils
                    quat = pose.quat  # [x, y, z, w]
                    quat_wxyz = np.array([quat[3], quat[0], quat[1], quat[2]])  # [w, x, y, z]
                    heading = float(arr_utils.quaternion_to_yaw(quat_wxyz))

                    # Velocity: Trajectory class doesn't have velocities_mps
                    # Velocities will be calculated by trajdata from positions
                    vx, vy = 0.0, 0.0

                    rows.append({
                        'agent_id': str(track_id),
                        'timestep': t_idx,
                        'x': x,
                        'y': y,
                        'z': z,
                        'vx': vx,
                        'vy': vy,
                        'heading': heading,
                        'length': float(traffic_obj.aabb.x) if traffic_obj.aabb else 4.5,
                        'width': float(traffic_obj.aabb.y) if traffic_obj.aabb else 2.0,
                        'agent_type': str(traffic_obj.label_class) if hasattr(traffic_obj, 'label_class') else 'vehicle',
                    })
            except Exception as e:
                logger.warning(f"Failed to process traffic object {track_id}: {e}")
                continue

        df = pd.DataFrame(rows)
        num_agents = len(df['agent_id'].unique()) if len(df) > 0 else 0
        logger.info(f"Converted {len(df)} trajectory points from {num_agents} agents")

        if len(df) == 0:
            return df

        # Compute velocities and accelerations in a single pass
        def compute_derivatives(group):
            """Compute velocity and acceleration for an agent."""
            if len(group) > 1:
                # Velocity from position differences
                group['vx'] = group['x'].diff().fillna(0.0) / dt
                group['vy'] = group['y'].diff().fillna(0.0) / dt
                # Acceleration from velocity differences
                group['ax'] = group['vx'].diff().fillna(0.0) / dt
                group['ay'] = group['vy'].diff().fillna(0.0) / dt
            else:
                # Single frame: zero velocity and acceleration
                group[['vx', 'vy', 'ax', 'ay']] = 0.0
            return group

        df = df.groupby('agent_id', group_keys=False).apply(compute_derivatives)

        logger.info(f"Computed velocities and accelerations for {num_agents} agents (dt={dt:.4f}s)")
        return df

    def get_agent_info(
        self, scene: Scene, cache_path: Path, cache_class: Type
    ):
        """
        Extract agent metadata from scene and cache agent data.

        This method is required by trajdata's RawDataset interface.

        Args:
            scene: Scene object containing agent_data DataFrame
            cache_path: Path to cache directory
            cache_class: SceneCache class for saving data

        Returns:
            Tuple of (agent_list, agent_presence)
        """
        from trajdata.data_structures.agent import AgentMetadata, AgentType, FixedExtent

        # Get the DataFrame we created in get_scene()
        scene_data = scene.agent_data

        if scene_data is None or len(scene_data) == 0:
            logger.warning(f"No agent data for scene {scene.name}")
            return [], [[] for _ in range(scene.length_timesteps)]

        # Ensure required columns exist
        required_cols = ['agent_id', 'timestep', 'x', 'y']
        if not all(col in scene_data.columns for col in required_cols):
            logger.error(f"Missing required columns in agent_data: {scene_data.columns}")
            return [], [[] for _ in range(scene.length_timesteps)]

        # Convert to proper format for trajdata
        # Rename 'timestep' to 'scene_ts' if needed
        if 'scene_ts' not in scene_data.columns:
            scene_data = scene_data.rename(columns={'timestep': 'scene_ts'})

        # Set index for efficient grouping
        scene_data = scene_data.set_index(['agent_id', 'scene_ts']).sort_index()
        scene_data = scene_data.reset_index(level=1)  # Keep scene_ts as column

        # Build agent metadata
        agent_list = []
        agent_presence = [[] for _ in range(scene.length_timesteps)]

        for agent_id, frames in scene_data.groupby(level=0)['scene_ts']:
            if len(frames) == 0:
                continue

            start_frame = int(frames.iloc[0])
            last_frame = int(frames.iloc[-1])

            # Get agent type and extent from the data
            agent_rows = scene_data.loc[agent_id]
            if isinstance(agent_rows, pd.Series):
                agent_rows = agent_rows.to_frame().T

            agent_type_str = agent_rows['agent_type'].iloc[0] if 'agent_type' in agent_rows.columns else 'vehicle'
            length = agent_rows['length'].iloc[0] if 'length' in agent_rows.columns else 4.5
            width = agent_rows['width'].iloc[0] if 'width' in agent_rows.columns else 2.0
            height = 1.5  # Default vehicle height

            # Map agent type string to AgentType enum
            if agent_type_str == 'ego' or agent_id == 'ego':
                agent_type = AgentType.VEHICLE
            elif 'vehicle' in str(agent_type_str).lower():
                agent_type = AgentType.VEHICLE
            elif 'pedestrian' in str(agent_type_str).lower():
                agent_type = AgentType.PEDESTRIAN
            else:
                agent_type = AgentType.UNKNOWN

            agent_metadata = AgentMetadata(
                name=str(agent_id),
                agent_type=agent_type,
                first_timestep=start_frame,
                last_timestep=last_frame,
                extent=FixedExtent(length, width, height),
            )

            agent_list.append(agent_metadata)

            # Add to agent_presence for each frame
            for frame in frames:
                if 0 <= frame < scene.length_timesteps:
                    agent_presence[int(frame)].append(agent_metadata)

        # Reset index for caching
        scene_data = scene_data.reset_index()
        scene_data = scene_data.set_index(['agent_id', 'scene_ts'])

        # Cache the agent data
        cache_class.save_agent_data(scene_data, cache_path, scene)

        logger.info(f"Cached {len(agent_list)} agents for scene {scene.name}")
        return agent_list, agent_presence

    def cache_all_scenes_list(self, cache_path: Path, verbose: bool = False) -> None:
        """Cache scene records list."""
        import pickle
        scenes_list_path = cache_path / "scenes_list.dill"
        with open(scenes_list_path, 'wb') as f:
            pickle.dump(self._scene_records, f)
        if verbose:
            logger.info(f"Cached {len(self._scene_records)} scene records")

    def cache_maps(self, cache_path: Path, map_cache_class: Type, verbose: bool = False) -> None:
        """Maps are cached automatically during get_scene()."""
        pass
