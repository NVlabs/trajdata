"""
Generic CSV Dataset adapter for trajdata.

Each CSV file represents one scene and must contain at minimum:
  frame_id, agent_id, x, y

Optional columns (computed from x/y if missing):
  vx, vy, heading, agent_type

CSV files are placed in a single directory; an optional ``config.json``
specifies the time-step and train/val/test splits.

Directory layout::

    /path/to/csv_data/
        ├── config.json          (optional)
        ├── scene_001.csv
        ├── scene_002.csv
        └── ...

``config.json`` example::

    {
        "dt": 0.1,
        "splits": {
            "train": ["scene_001", "scene_002"],
            "val":   ["scene_003"]
        }
    }

Register in UnifiedDataset as ``csv_<name>``, e.g.::

    dataset = UnifiedDataset(
        desired_data=["csv_mydata-train"],
        data_dirs={"csv_mydata": "/path/to/csv_data"},
    )
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Tuple, Type

import numpy as np
import pandas as pd

from trajdata.caching import EnvCache, SceneCache
from trajdata.data_structures.agent import AgentMetadata, AgentType, FixedExtent
from trajdata.data_structures.environment import EnvMetadata
from trajdata.data_structures.scene_metadata import Scene, SceneMetadata
from trajdata.data_structures.scene_tag import SceneTag
from trajdata.dataset_specific.raw_dataset import RawDataset
from trajdata.dataset_specific.scene_records import EUPedsRecord
from trajdata.utils import arr_utils

logger = logging.getLogger(__name__)

DEFAULT_DT: Final[float] = 0.1
DEFAULT_EXTENT: Final[Tuple[float, float, float]] = (0.5, 0.5, 1.7)

_AGENT_TYPE_MAP: Final[Dict[str, AgentType]] = {
    "pedestrian": AgentType.PEDESTRIAN,
    "ped": AgentType.PEDESTRIAN,
    "vehicle": AgentType.VEHICLE,
    "car": AgentType.VEHICLE,
    "bicycle": AgentType.BICYCLE,
    "bike": AgentType.BICYCLE,
    "motorcycle": AgentType.MOTORCYCLE,
}


def _parse_agent_type(value) -> AgentType:
    if isinstance(value, str):
        return _AGENT_TYPE_MAP.get(value.lower(), AgentType.UNKNOWN)
    return AgentType.UNKNOWN


class CSVDataset(RawDataset):
    """Adapter that loads any directory of per-scene CSV files."""

    def compute_metadata(self, env_name: str, data_dir: str) -> EnvMetadata:
        data_dir = Path(data_dir).expanduser()
        config = self._load_config(data_dir)

        self._dt: float = config.get("dt", DEFAULT_DT)
        self._config_splits: Dict[str, List[str]] = config.get("splits", {})

        dataset_parts: List[Tuple[str, ...]] = [("train", "val", "test")]
        scene_split_map: Dict[str, str] = {}

        # Assign each CSV scene to a split using config; default to "train"
        for csv_path in sorted(data_dir.glob("*.csv")):
            scene_name = csv_path.stem
            assigned = "train"
            for split, names in self._config_splits.items():
                if scene_name in names:
                    assigned = split
                    break
            scene_split_map[scene_name] = assigned

        self._scene_split_map = scene_split_map

        return EnvMetadata(
            name=env_name,
            data_dir=str(data_dir),
            dt=self._dt,
            parts=dataset_parts,
            scene_split_map=scene_split_map,
        )

    # ------------------------------------------------------------------
    # Dataset object loading
    # ------------------------------------------------------------------

    def load_dataset_obj(self, verbose: bool = False) -> None:
        if verbose:
            logger.info("Loading CSV dataset from %s ...", self.metadata.data_dir)

        data_dir = Path(self.metadata.data_dir)
        self.dataset_obj: Dict[str, pd.DataFrame] = {}

        for csv_path in sorted(data_dir.glob("*.csv")):
            scene_name = csv_path.stem
            df = pd.read_csv(csv_path)
            df = self._normalise_columns(df)
            df["frame_id"] = pd.to_numeric(
                df["frame_id"] - df["frame_id"].min(), downcast="integer"
            )
            self.dataset_obj[scene_name] = df

    # ------------------------------------------------------------------
    # Scene discovery helpers
    # ------------------------------------------------------------------

    def _get_matching_scenes_from_obj(
        self,
        scene_tag: SceneTag,
        scene_desc_contains: Optional[List[str]],
        env_cache: EnvCache,
    ) -> List[SceneMetadata]:
        all_records: List[EUPedsRecord] = []
        scenes: List[SceneMetadata] = []

        for idx, (scene_name, df) in enumerate(self.dataset_obj.items()):
            split = self.metadata.scene_split_map.get(scene_name, "train")
            length = int(df["frame_id"].max()) + 1

            all_records.append(
                EUPedsRecord(scene_name, "csv", length, split, idx)
            )

            if split in scene_tag and (
                scene_desc_contains is None
                or any(k in scene_name for k in scene_desc_contains)
            ):
                scenes.append(
                    SceneMetadata(
                        env_name=self.metadata.name,
                        name=scene_name,
                        dt=self.metadata.dt,
                        raw_data_idx=idx,
                    )
                )

        self.cache_all_scenes_list(env_cache, all_records)
        return scenes

    def _get_matching_scenes_from_cache(
        self,
        scene_tag: SceneTag,
        scene_desc_contains: Optional[List[str]],
        env_cache: EnvCache,
    ) -> List[Scene]:
        all_records: List[EUPedsRecord] = env_cache.load_env_scenes_list(self.name)
        scenes: List[Scene] = []

        for record in all_records:
            scene_name, _loc, length, split, data_idx = record
            if split in scene_tag and (
                scene_desc_contains is None
                or any(k in scene_name for k in scene_desc_contains)
            ):
                scenes.append(
                    Scene(
                        self.metadata,
                        scene_name,
                        "csv",
                        split,
                        length,
                        data_idx,
                        None,
                    )
                )
        return scenes

    def get_scene(self, scene_info: SceneMetadata) -> Scene:
        _, scene_name, _, data_idx = scene_info
        df = self.dataset_obj[scene_name]
        split = self.metadata.scene_split_map.get(scene_name, "train")
        length = int(df["frame_id"].max()) + 1
        return Scene(self.metadata, scene_name, "csv", split, length, data_idx, None)

    # ------------------------------------------------------------------
    # Agent info extraction
    # ------------------------------------------------------------------

    def get_agent_info(
        self, scene: Scene, cache_path: Path, cache_class: Type[SceneCache]
    ) -> Tuple[List[AgentMetadata], List[List[AgentMetadata]]]:
        df = self.dataset_obj[scene.name].copy()
        df.rename(
            columns={"frame_id": "scene_ts", "agent_id": "agent_id"},
            inplace=True,
        )
        df["agent_id"] = df["agent_id"].astype(str)
        df.set_index(["agent_id", "scene_ts"], inplace=True)
        df.sort_index(inplace=True)
        df.reset_index(level=1, inplace=True)

        agent_ids = df.index.get_level_values(0).to_numpy()

        # z column
        if "z" not in df.columns:
            df["z"] = 0.0

        # velocities
        if "vx" not in df.columns or "vy" not in df.columns:
            vel = (
                arr_utils.agent_aware_diff(df[["x", "y"]].to_numpy(), agent_ids)
                / self.metadata.dt
            )
            df["vx"], df["vy"] = vel[:, 0], vel[:, 1]

        # accelerations
        if "ax" not in df.columns or "ay" not in df.columns:
            acc = (
                arr_utils.agent_aware_diff(df[["vx", "vy"]].to_numpy(), agent_ids)
                / self.metadata.dt
            )
            df["ax"], df["ay"] = acc[:, 0], acc[:, 1]

        # heading
        if "heading" not in df.columns:
            df["heading"] = np.arctan2(df["vy"], df["vx"])

        # agent_type per-row
        if "agent_type" in df.columns:
            df["_atype"] = df["agent_type"].apply(_parse_agent_type)
        else:
            df["_atype"] = AgentType.PEDESTRIAN

        # Build metadata lists
        agent_list: List[AgentMetadata] = []
        agent_presence: List[List[AgentMetadata]] = [
            [] for _ in range(scene.length_timesteps)
        ]

        for agent_id, frames_df in df.groupby(level=0):
            frames = frames_df["scene_ts"]
            if len(frames) <= 1:
                continue

            # pandas may store enum as int; cast back to AgentType
            raw_atype = frames_df["_atype"].iat[0]
            atype = AgentType(int(raw_atype)) if not isinstance(raw_atype, AgentType) else raw_atype
            t0, t1 = int(frames.iat[0]), int(frames.iat[-1])
            meta = AgentMetadata(
                name=str(agent_id),
                agent_type=atype,
                first_timestep=t0,
                last_timestep=t1,
                extent=FixedExtent(*DEFAULT_EXTENT),
            )
            agent_list.append(meta)
            for ts in frames:
                if 0 <= ts < scene.length_timesteps:
                    agent_presence[ts].append(meta)

        # Drop helper columns before caching
        df.drop(columns=["_atype"], inplace=True, errors="ignore")
        df.drop(columns=["agent_type"], inplace=True, errors="ignore")

        # Restore (agent_id, scene_ts) MultiIndex expected by save_agent_data
        df.reset_index(inplace=True)  # agent_id becomes column again
        df["agent_id"] = df["agent_id"].astype(str)
        df.set_index(["agent_id", "scene_ts"], inplace=True)

        cache_class.save_agent_data(df, cache_path, scene)
        return agent_list, agent_presence

    # ------------------------------------------------------------------
    # Maps (none for CSV)
    # ------------------------------------------------------------------

    def cache_map(self, map_name, layer_names, cache_path, map_cache_class, resolution):
        pass

    def cache_maps(self, cache_path, map_cache_class, map_params):
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_config(data_dir: Path) -> dict:
        cfg_path = data_dir / "config.json"
        if cfg_path.exists():
            with open(cfg_path) as f:
                return json.load(f)
        return {}

    @staticmethod
    def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Rename common column aliases to canonical names."""
        rename = {}
        lc = {c.lower(): c for c in df.columns}
        for canonical, aliases in {
            "frame_id": ["frame_id", "frame", "timestep", "t"],
            "agent_id": ["agent_id", "track_id", "id", "object_id"],
            "x": ["x", "pos_x", "position_x"],
            "y": ["y", "pos_y", "position_y"],
        }.items():
            for alias in aliases:
                if alias in lc and canonical not in df.columns:
                    rename[lc[alias]] = canonical
                    break
        return df.rename(columns=rename)
