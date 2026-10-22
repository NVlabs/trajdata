import re
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, Final, List, NamedTuple, Optional, Tuple, Type

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

from trajdata.caching import EnvCache, SceneCache
from trajdata.data_structures.agent import AgentMetadata, AgentType, FixedExtent
from trajdata.data_structures.environment import EnvMetadata
from trajdata.data_structures.scene_metadata import Scene, SceneMetadata
from trajdata.data_structures.scene_tag import SceneTag
from trajdata.dataset_specific.raw_dataset import RawDataset
from trajdata.dataset_specific.scene_records import SongdoSceneRecord

# Songdo Traffic was captured at 29.97 fps (NTSC 30000/1001). As with the Stanford Drone
# Dataset, we declare the nominal 30 Hz, so that trajdata's timestep interpolation can
# reach the 0.1s most other datasets here use. Positions and the published speeds are
# unaffected; the cost is that a timeline reported as 1/30 s per step really advances by
# 1001/30000 s, so differentiating the positions against this dt overstates speed by
# 0.1% relative to the published value.
SONGDO_DT: Final[float] = 1.0 / 30.0

# The true inter-frame period, used only to recover frame indices from the recorded
# wall-clock timestamps (which are rounded to milliseconds and jitter by a frame
# period).
SONGDO_FRAME_PERIOD: Final[float] = 1001.0 / 30000.0

# Each aggregated CSV covers one flight session, during which a drone takes off, records
# a clip of a few minutes, and lands to swap batteries. A break longer than this splits
# the session into separate scenes. The choice is not delicate: over a sample of 80
# sessions every threshold from 10s to 60s recovers exactly the same 296 clips, because
# the breaks between clips all run to minutes while an intersection briefly empty of
# vehicles goes quiet for only a few seconds. Below 10s the lulls start splitting clips.
SONGDO_CLIP_GAP_S: Final[float] = 10.0

# Scenes shorter than this are dropped, being too short to sample a history/future
# window.
SONGDO_MIN_SCENE_LEN: Final[int] = 60

# The dataset publishes speed and acceleration as scalars but no heading, so the
# direction of travel is the one kinematic quantity that has to be derived here. These
# reproduce geo-trax's own heading estimator (visualize.py, compute_headings): the
# position differences are smoothed with a Gaussian of this standard deviation [frames],
# and frames whose smoothed speed falls below the threshold have no observable direction
# and inherit the nearest reliable heading instead. geo-trax's default threshold of
# 0.5 px/frame becomes 0.41 m/s at this data's 0.02725 m/px ground sampling distance and
# 29.97 fps.
SONGDO_HEADING_SIGMA: Final[float] = 15.0
SONGDO_HEADING_MIN_SPEED: Final[float] = 0.41

# km/h, as the dataset publishes speed, to m/s, as trajdata expects it.
SONGDO_KMH_TO_MS: Final[float] = 1.0 / 3.6

# The 20 intersections covered by the experiment, labelled A-U in the dataset (D is
# unused).
SONGDO_INTERSECTIONS: Final[Tuple[str, ...]] = tuple(
    f"int_{letter.lower()}" for letter in "ABCEFGHIJKLMNOPQRSTU"
)

# The dataset ships without an official split, so we hold out whole days: this keeps all
# 20 intersections in every split and rules out the same vehicle appearing on both sides
# of it.
SONGDO_DATE_SPLITS: Final[Dict[str, str]] = {
    "2022-10-04": "train",
    "2022-10-05": "train",
    "2022-10-06": "val",
    "2022-10-07": "test",
}

# Vehicle_Class values, as documented in the dataset's own README.
SONGDO_AGENT_TYPES: Final[Dict[int, AgentType]] = {
    0: AgentType.VEHICLE,  # car (including vans)
    1: AgentType.VEHICLE,  # bus
    2: AgentType.VEHICLE,  # truck
    3: AgentType.MOTORCYCLE,
}

# Fallbacks for the few percent of vehicles whose dimensions were never estimated. The
# lengths and widths are the per-class medians of the published estimates over fully
# visible observations; heights are not estimated by the dataset at all, so those are
# representative values for the vehicle fleet.
SONGDO_DEFAULT_EXTENTS: Final[Dict[int, Tuple[float, float, float]]] = {
    0: (4.91, 2.16, 1.55),
    1: (10.29, 2.95, 3.20),
    2: (6.07, 2.37, 2.90),
    3: (2.37, 1.09, 1.60),
}

SONGDO_CSV_RE: Final[re.Pattern] = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})_(?P<loc>[A-U])_(?P<session>(AM|PM)\d)\.csv$"
)
SONGDO_ZIP_RE: Final[re.Pattern] = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})_(?P<loc>[A-U])\.zip$"
)

SONGDO_USED_COLUMNS: Final[List[str]] = [
    "Vehicle_ID",
    "Local_Time",
    "Drone_ID",
    "Local_X",
    "Local_Y",
    "Vehicle_Length",
    "Vehicle_Width",
    "Vehicle_Class",
    "Vehicle_Speed",
    "Vehicle_Acceleration",
]


# What get_agent_info returns: the scene's agents, and who is present at each timestep.
AgentInfo = Tuple[List[AgentMetadata], List[List[AgentMetadata]]]


class SongdoBlock(NamedTuple):
    """One contiguous single-drone recording within an aggregated session CSV."""

    source: Path  # The .csv on disk, or the .zip holding it.
    member: Optional[str]  # The CSV's name inside source, when source is a .zip.
    drone_id: int
    first_frame: int  # Index of this block's first frame within the session.
    length: int
    location: str
    split: str
    origin: Tuple[float, float]  # EPSG:5186 easting/northing subtracted from positions.
    # Position in the dataset's scene list, assigned once every block is known.
    data_idx: int = -1


def songdo_location(letter: str) -> str:
    return f"int_{letter.lower()}"


def songdo_session_period(session: str) -> str:
    return session[:2].lower()


def read_songdo_csv(source: Path, member: Optional[str], **kwargs: Any) -> pd.DataFrame:
    """Read one session CSV, either standalone or from inside its Zenodo archive."""
    if member is None:
        return pd.read_csv(source, **kwargs)

    with zipfile.ZipFile(source) as archive, archive.open(member) as csv_file:
        return pd.read_csv(csv_file, **kwargs)


def frame_indices(timestamps: np.ndarray) -> Dict[float, int]:
    """Map each distinct timestamp of a session to its video frame index.

    The timestamps are wall-clock times rounded to milliseconds, so they drift
    against an exact multiple of the frame period and cannot simply be divided by
    it. Accumulating the rounded gaps instead is exact, and leaves a hole wherever
    a frame held no vehicle at all.
    """
    unique_ts: np.ndarray = np.unique(timestamps)
    steps: np.ndarray = np.maximum(
        1, np.round(np.diff(unique_ts) / SONGDO_FRAME_PERIOD).astype(np.int64)
    )
    indices: np.ndarray = np.concatenate(([0], np.cumsum(steps)))
    return dict(zip(unique_ts.tolist(), indices.tolist()))


def session_frames(session_df: pd.DataFrame) -> pd.Series:
    """Frame index of every row of a session, on its own drone's clock.

    Each drone times its own recording, so two drones covering one intersection carry
    unrelated clocks whose millisecond timestamps interleave. Resolving the frames of a
    whole session at once would read that interleaving as a doubled frame rate, so each
    drone is resolved separately and numbered from its own first frame.
    """
    seconds: pd.Series = pd.to_timedelta(session_df["Local_Time"]).dt.total_seconds()

    frames: pd.Series = pd.Series(index=session_df.index, dtype="int64")
    for _, drone_seconds in seconds.groupby(session_df["Drone_ID"]):
        frames.loc[drone_seconds.index] = drone_seconds.map(
            frame_indices(drone_seconds.to_numpy())
        ).astype("int64")

    return frames


def split_into_blocks(frames: np.ndarray) -> List[Tuple[int, int]]:
    """Split one drone's frame indices into the clips it actually recorded."""
    unique_frames: np.ndarray = np.unique(frames)
    gap_frames: int = int(round(SONGDO_CLIP_GAP_S / SONGDO_FRAME_PERIOD))
    breaks: np.ndarray = np.nonzero(np.diff(unique_frames) > gap_frames)[0]

    starts: np.ndarray = np.concatenate(([0], breaks + 1))
    ends: np.ndarray = np.concatenate((breaks, [len(unique_frames) - 1]))
    return [
        (int(unique_frames[start]), int(unique_frames[end]))
        for start, end in zip(starts, ends)
    ]


def agent_headings(positions: np.ndarray) -> np.ndarray:
    """Heading [rad] of one agent, from the direction its position moves in.

    The dataset publishes no heading, so it is estimated the way geo-trax itself does
    (``visualize.py``, ``compute_headings``): the position differences are smoothed with
    a Gaussian and the heading is their arctangent. A vehicle slower than the threshold
    has no observable direction and inherits the nearest reliable heading, held rather
    than interpolated, since angles do not average across the wraparound. Over half the
    observations in a clip can be of vehicles queueing at a signal or parked.
    """
    if len(positions) < 2:
        return np.zeros(len(positions))

    deltas: np.ndarray = gaussian_filter1d(
        np.gradient(positions, axis=0), SONGDO_HEADING_SIGMA, axis=0, mode="reflect"
    )
    # The gradient is per frame, so convert to m/s before applying the speed threshold.
    reliable: np.ndarray = (
        np.linalg.norm(deltas, axis=1) / SONGDO_DT >= SONGDO_HEADING_MIN_SPEED
    )
    if not reliable.any():
        # Never seen moving, so fall back to the direction it drifted in overall. This
        # is arbitrary for a parked vehicle, which has no direction of travel at all.
        drift: np.ndarray = positions[-1] - positions[0]
        return np.full(len(positions), np.arctan2(drift[1], drift[0]))

    headings: np.ndarray = np.where(
        reliable, np.arctan2(deltas[:, 1], deltas[:, 0]), np.nan
    )
    return pd.Series(headings).ffill().bfill().to_numpy()


class SongdoDataset(RawDataset):
    def compute_metadata(self, env_name: str, data_dir: str) -> EnvMetadata:
        # Songdo Traffic possibilities are the Cartesian product of these.
        dataset_parts: List[Tuple[str, ...]] = [
            ("train", "val", "test"),
            SONGDO_INTERSECTIONS,
            ("am", "pm"),
        ]

        return EnvMetadata(
            name=env_name,
            data_dir=data_dir,
            dt=SONGDO_DT,
            parts=dataset_parts,
            # The split follows from the date in the scene name, see SONGDO_DATE_SPLITS.
            scene_split_map=None,
        )

    def _session_files(self) -> Dict[Tuple[str, str, str], Tuple[Path, Optional[str]]]:
        """Locate every session CSV below the data directory.

        Both layouts Zenodo offers are accepted: the 80 ``<date>_<intersection>.zip``
        archives as downloaded, or their CSVs extracted anywhere below the data
        directory. Extracting all of them costs about 165 GB, so reading from the
        archives is usually the better trade; an extracted CSV takes precedence
        over an archived one.
        """
        data_dir: Path = Path(self.metadata.data_dir)
        sessions: Dict[Tuple[str, str, str], Tuple[Path, Optional[str]]] = dict()

        for archive_path in sorted(data_dir.glob("**/*.zip")):
            if SONGDO_ZIP_RE.match(archive_path.name) is None:
                # The record also ships orthophotos, master frames and sample videos.
                continue

            with zipfile.ZipFile(archive_path) as archive:
                members: List[str] = archive.namelist()

            for member in members:
                match = SONGDO_CSV_RE.match(Path(member).name)
                if match is not None:
                    key = (match["date"], match["loc"], match["session"])
                    sessions[key] = (archive_path, member)

        for csv_path in sorted(data_dir.glob("**/*.csv")):
            match = SONGDO_CSV_RE.match(csv_path.name)
            if match is not None:
                key = (match["date"], match["loc"], match["session"])
                sessions[key] = (csv_path, None)

        return sessions

    def load_dataset_obj(self, verbose: bool = False) -> None:
        if verbose:
            print(f"Loading {self.name} dataset...", flush=True)

        sessions = self._session_files()
        if not sessions:
            raise ValueError(
                f"No Songdo Traffic sessions found under {self.metadata.data_dir}. "
                "Expected the <date>_<intersection>.zip archives from the Zenodo "
                "record, or their extracted "
                "<date>_<intersection>_<session>.csv files."
            )

        # Only the timestamps and positions are needed to lay out the scenes; the
        # vehicle data itself is read later, once per session, in get_agent_info.
        self.dataset_obj: Dict[str, SongdoBlock] = dict()
        for (date, letter, session), (source, member) in tqdm(
            sorted(sessions.items()),
            desc=f"Indexing {self.name} sessions",
            disable=not verbose,
        ):
            session_df: pd.DataFrame = read_songdo_csv(
                source,
                member,
                usecols=["Local_Time", "Drone_ID", "Local_X", "Local_Y"],
            )
            session_df["frame"] = session_frames(session_df)

            for drone_id, drone_df in session_df.groupby("Drone_ID"):
                for block_num, (first, last) in enumerate(
                    split_into_blocks(drone_df["frame"].to_numpy())
                ):
                    length: int = last - first + 1
                    if length < SONGDO_MIN_SCENE_LEN:
                        continue

                    block_df = drone_df[drone_df["frame"].between(first, last)]
                    scene_name: str = (
                        f"{date}_{letter}_{session}_D{drone_id}_{block_num}"
                    )
                    self.dataset_obj[scene_name] = SongdoBlock(
                        source=source,
                        member=member,
                        drone_id=int(drone_id),
                        first_frame=first,
                        length=length,
                        location=songdo_location(letter),
                        split=SONGDO_DATE_SPLITS[date],
                        # Positions are recentred so that they survive the conversion to
                        # float32 downstream: EPSG:5186 northings near 533000 m only
                        # resolve to about 6cm in float32, which would visibly quantise
                        # them.
                        origin=(
                            round(float(block_df["Local_X"].mean()), 1),
                            round(float(block_df["Local_Y"].mean()), 1),
                        ),
                    )

        self.dataset_obj = {
            scene_name: block._replace(data_idx=idx)
            for idx, (scene_name, block) in enumerate(self.dataset_obj.items())
        }

        if verbose:
            print(
                f"Found {len(self.dataset_obj)} Songdo Traffic scenes in "
                f"{len(sessions)} session files.",
                flush=True,
            )

    def _scene_description(self, scene_name: str) -> str:
        block: SongdoBlock = self.dataset_obj[scene_name]
        easting, northing = block.origin
        return (
            f"Drone {block.drone_id} over Songdo intersection "
            f"{block.location[-1].upper()}; positions are relative to EPSG:5186 "
            f"({easting:.1f}, {northing:.1f})"
        )

    def _get_matching_scenes_from_obj(
        self,
        scene_tag: SceneTag,
        scene_desc_contains: Optional[List[str]],
        env_cache: EnvCache,
    ) -> List[SceneMetadata]:
        all_scenes_list: List[SongdoSceneRecord] = list()

        scenes_list: List[SceneMetadata] = list()
        for scene_name, block in self.dataset_obj.items():
            description: str = self._scene_description(scene_name)

            # Saving all scene records for later caching.
            all_scenes_list.append(
                SongdoSceneRecord(
                    scene_name,
                    block.location,
                    block.length,
                    description,
                    block.data_idx,
                )
            )

            if (
                block.location in scene_tag
                and block.split in scene_tag
                and songdo_session_period(scene_name.split("_")[2]) in scene_tag
                and (
                    scene_desc_contains is None
                    or any(desc in description for desc in scene_desc_contains)
                )
            ):
                scenes_list.append(
                    SceneMetadata(
                        env_name=self.metadata.name,
                        name=scene_name,
                        dt=self.metadata.dt,
                        raw_data_idx=block.data_idx,
                    )
                )

        self.cache_all_scenes_list(env_cache, all_scenes_list)
        return scenes_list

    def _get_matching_scenes_from_cache(
        self,
        scene_tag: SceneTag,
        scene_desc_contains: Optional[List[str]],
        env_cache: EnvCache,
    ) -> List[Scene]:
        all_scenes_list: List[SongdoSceneRecord] = env_cache.load_env_scenes_list(
            self.name
        )

        scenes_list: List[Scene] = list()
        for scene_record in all_scenes_list:
            scene_name, scene_location, scene_length, description, data_idx = (
                scene_record
            )
            scene_split: str = SONGDO_DATE_SPLITS[scene_name.split("_")[0]]

            if (
                scene_location in scene_tag
                and scene_split in scene_tag
                and songdo_session_period(scene_name.split("_")[2]) in scene_tag
                and (
                    scene_desc_contains is None
                    or any(desc in description for desc in scene_desc_contains)
                )
            ):
                scenes_list.append(
                    Scene(
                        self.metadata,
                        scene_name,
                        scene_location,
                        scene_split,
                        scene_length,
                        data_idx,
                        None,  # This isn't used if everything is already cached.
                        description,
                    )
                )

        return scenes_list

    def get_scene(self, scene_info: SceneMetadata) -> Scene:
        _, scene_name, _, data_idx = scene_info
        block: SongdoBlock = self.dataset_obj[scene_name]

        return Scene(
            self.metadata,
            scene_name,
            block.location,
            block.split,
            block.length,
            data_idx,
            None,  # No data access info necessary for the Songdo Traffic dataset.
            self._scene_description(scene_name),
        )

    def _sibling_scenes(self, scene_name: str) -> List[str]:
        """The scenes drawn from the same session CSV as scene_name, itself included."""
        block: SongdoBlock = self.dataset_obj[scene_name]
        return [
            name
            for name, other in self.dataset_obj.items()
            if other.source == block.source and other.member == block.member
        ]

    def get_agent_info(
        self, scene: Scene, cache_path: Path, cache_class: Type[SceneCache]
    ) -> AgentInfo:
        scene_metadata_path = EnvCache.scene_metadata_path(
            cache_path, scene.env_name, scene.name, scene.dt
        )
        if scene_metadata_path.exists():
            # Try repeatedly to open the file because it might still be being created in
            # another process.
            while True:
                try:
                    already_done_scene = EnvCache.load(scene_metadata_path)
                    break
                except:
                    time.sleep(1)

            # Already processed, so we can immediately return our cached results.
            return already_done_scene.agents, already_done_scene.agent_presence

        block: SongdoBlock = self.dataset_obj[scene.name]
        session_df: pd.DataFrame = read_songdo_csv(
            block.source, block.member, usecols=SONGDO_USED_COLUMNS
        )
        session_df["frame"] = session_frames(session_df)

        # One session CSV holds a handful of clips and takes a few seconds to read, so
        # every scene drawn from this one is processed and cached here rather than
        # re-reading it.
        results: Dict[str, AgentInfo] = dict()
        for scene_name in self._sibling_scenes(scene.name):
            sibling: SongdoBlock = self.dataset_obj[scene_name]
            block_df: pd.DataFrame = session_df[
                (session_df["Drone_ID"] == sibling.drone_id)
                & session_df["frame"].between(
                    sibling.first_frame, sibling.first_frame + sibling.length - 1
                )
            ]
            block_scene = Scene(
                env_metadata=scene.env_metadata,
                name=scene_name,
                location=sibling.location,
                data_split=sibling.split,
                length_timesteps=sibling.length,
                raw_data_idx=sibling.data_idx,
                data_access_info=None,
                description=self._scene_description(scene_name),
            )
            results[scene_name] = self._cache_block(
                block_df, sibling, block_scene, cache_path, cache_class
            )

        return results[scene.name]

    def _cache_block(
        self,
        block_df: pd.DataFrame,
        block: SongdoBlock,
        scene: Scene,
        cache_path: Path,
        cache_class: Type[SceneCache],
    ) -> AgentInfo:
        # Speed and acceleration are the dataset's own published values, not recomputed
        # from the positions. They are absent for the partially visible observations
        # (Visibility == 0, about 5% of rows), which the dataset excludes from its
        # kinematics because a bounding box clipped by the frame edge moves its own
        # centre as the vehicle crosses it. Those rows keep their positions and are
        # passed through with empty kinematics rather than dropped or reconstructed.
        data_df: pd.DataFrame = pd.DataFrame(
            {
                "agent_id": block_df["Vehicle_ID"].astype(str),
                "scene_ts": block_df["frame"] - block.first_frame,
                "x": block_df["Local_X"] - block.origin[0],
                "y": block_df["Local_Y"] - block.origin[1],
                "z": 0.0,
                "speed": block_df["Vehicle_Speed"] * SONGDO_KMH_TO_MS,
                "accel": block_df["Vehicle_Acceleration"],
            }
        )

        # Class and dimensions are published per vehicle, not per observation.
        classes: pd.Series = block_df.groupby(block_df["Vehicle_ID"].astype(str))[
            "Vehicle_Class"
        ].first()
        extents: pd.DataFrame = self._agent_extents(block_df, classes)

        # Ensuring data is sorted by agent ID and scene timestep.
        data_df.set_index(["agent_id", "scene_ts"], inplace=True)
        data_df.sort_index(inplace=True)

        # A track skips a frame wherever the detector lost the vehicle, which is about
        # 0.05% of timesteps. trajdata addresses an agent's state by exact timestep, so
        # those holes have to be filled; positions are interpolated linearly, and the
        # published speed and acceleration are left empty rather than invented.
        data_df = self._densify(data_df)

        agent_ids: np.ndarray = data_df.index.get_level_values(0).to_numpy()
        positions: np.ndarray = data_df[["x", "y"]].to_numpy()

        headings: np.ndarray = np.empty(len(positions))
        for start, stop in _agent_slices(agent_ids):
            headings[start:stop] = agent_headings(positions[start:stop])

        # The dataset publishes speed and acceleration as magnitudes along the direction
        # of travel, so they become vectors by rotating them into the heading. Both stay
        # empty wherever the dataset leaves them empty, which is every observation of a
        # vehicle only partly inside the frame: a clipped bounding box moves its centre
        # as the vehicle enters or leaves, and the dataset deliberately derives no
        # kinematics from it. Note that the acceleration this yields is tangential only,
        # the centripetal component not being recoverable from a published magnitude.
        direction: np.ndarray = np.stack((np.cos(headings), np.sin(headings)), axis=1)
        data_df[["vx", "vy"]] = data_df[["speed"]].to_numpy() * direction
        data_df[["ax", "ay"]] = data_df[["accel"]].to_numpy() * direction
        data_df["heading"] = headings

        data_df.reset_index(level=1, inplace=True)

        agent_list: List[AgentMetadata] = list()
        agent_presence: List[List[AgentMetadata]] = [
            [] for _ in range(scene.length_timesteps)
        ]
        for agent_id, frames in data_df.groupby(level=0)["scene_ts"]:
            length, width, height = extents.loc[agent_id]
            agent_metadata = AgentMetadata(
                name=agent_id,
                agent_type=SONGDO_AGENT_TYPES[int(classes.loc[agent_id])],
                first_timestep=int(frames.iat[0]),
                last_timestep=int(frames.iat[-1]),
                extent=FixedExtent(float(length), float(width), float(height)),
            )

            agent_list.append(agent_metadata)
            for frame in frames:
                agent_presence[frame].append(agent_metadata)

        data_df.set_index("scene_ts", append=True, inplace=True)
        cache_class.save_agent_data(
            data_df.loc[:, ["x", "y", "z", "vx", "vy", "ax", "ay", "heading"]],
            cache_path,
            scene,
        )
        scene.update_agent_info(agent_list, agent_presence)
        EnvCache.save_scene_with_path(cache_path, scene)

        return agent_list, agent_presence

    @staticmethod
    def _agent_extents(block_df: pd.DataFrame, classes: pd.Series) -> pd.DataFrame:
        """Each vehicle's published length and width, with a per-class fallback.

        The dataset estimates dimensions once per vehicle rather than per observation,
        from the lower quartile of its bounding boxes after azimuth and aspect-ratio
        filtering, so the value is constant along a track and simply read off here. It
        is empty for the vehicles that filtering left without a usable box.
        """
        measured: pd.DataFrame = (
            block_df.groupby(block_df["Vehicle_ID"].astype(str))[
                ["Vehicle_Length", "Vehicle_Width"]
            ]
            .first()
            .reindex(classes.index)
        )
        defaults: pd.DataFrame = pd.DataFrame(
            [SONGDO_DEFAULT_EXTENTS[cls] for cls in classes],
            index=classes.index,
            columns=["length", "width", "height"],
        )

        defaults["length"] = measured["Vehicle_Length"].fillna(defaults["length"])
        defaults["width"] = measured["Vehicle_Width"].fillna(defaults["width"])
        return defaults

    @staticmethod
    def _densify(data_df: pd.DataFrame) -> pd.DataFrame:
        """Fill the frames each track skips, interpolating its position linearly.

        Only the position is filled. The dataset publishes no speed or acceleration for
        a frame it has no detection in, and none is invented here.
        """
        timesteps: pd.Series = pd.Series(
            data_df.index.get_level_values(1), index=data_df.index.get_level_values(0)
        )
        spans: pd.DataFrame = timesteps.groupby(level=0).agg(["min", "max", "count"])
        if (spans["max"] - spans["min"] + 1 == spans["count"]).all():
            return data_df

        dense_index: pd.MultiIndex = pd.MultiIndex.from_tuples(
            [
                (agent_id, ts)
                for agent_id, (first, last, _) in spans.iterrows()
                for ts in range(int(first), int(last) + 1)
            ],
            names=data_df.index.names,
        )
        dense_df: pd.DataFrame = data_df.reindex(dense_index)
        dense_df[["x", "y", "z"]] = (
            dense_df[["x", "y", "z"]]
            .interpolate(method="linear", axis="index", limit_area="inside")
            .fillna({"z": 0.0})
        )
        return dense_df

    def cache_map(
        self,
        map_name: str,
        layer_names: List[str],
        cache_path: Path,
        map_cache_class: Type[SceneCache],
        resolution: float,
    ) -> None:
        """
        No maps in this dataset!
        """
        pass

    def cache_maps(
        self,
        cache_path: Path,
        map_cache_class: Type[SceneCache],
        map_params: Dict[str, Any],
    ) -> None:
        """
        The release describes its road network only as per-lane polygons in orthophoto
        pixels, which is not enough to build a VectorMap, so this dataset has no maps.
        """
        pass


def _agent_slices(agent_ids: np.ndarray) -> List[Tuple[int, int]]:
    """Row ranges of each agent in an array sorted by agent ID."""
    boundaries: np.ndarray = np.concatenate(
        ([0], np.nonzero(agent_ids[1:] != agent_ids[:-1])[0] + 1, [len(agent_ids)])
    )
    return list(zip(boundaries[:-1].tolist(), boundaries[1:].tolist()))
