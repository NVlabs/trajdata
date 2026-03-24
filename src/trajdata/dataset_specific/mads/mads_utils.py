# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import multiprocessing
import json
import os
import warnings
from concurrent import futures
from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from trajdata.maps.vec_map import VectorMap
from trajdata.maps.vec_map_elements import Polyline, RoadLane
from trajdata.data_structures.agent import AgentType

MAX_POLYLINE_POINT_DIST = 2.0


def df_expand_json(df: pd.DataFrame) -> pd.DataFrame:
    """Expand nested/json-like columns into flattened dotted columns.

    Args:
        df: Input DataFrame with nested object columns.

    Returns:
        Expanded DataFrame where each nested column is normalized and prefixed.
    """
    # Use explicit string column names for static type checkers.
    columns = [str(col) for col in df.columns]
    for key in columns:
        cap_key = (
            "".join([x.capitalize() for x in key.split("_")])
            if key not in ["key", "version"]
            else key
        )
        df = df.join(pd.json_normalize(df.loc[:, key]).add_prefix(f"{cap_key}."))

    return df


# quaternion from direction
import math
def quaternion_from_direction(v):
    """
    Build a quaternion (w, x, y, z) that rotates (1,0,0) onto the direction of v.
    """

    x, y, _ = v
    # compute yaw angle
    yaw = math.atan2(y, x)
    half = yaw * 0.5

    w = math.cos(half)
    # rotation axis is Z, so only z component non‑zero
    return (0.0, 0.0, math.sin(half), w)

def add_quaternion_from_direction(df, x_col='x', y_col='y', z_col='z'):
    """
    Given a DataFrame with 3D vectors in columns x_col, y_col, z_col,
    compute for each row the quaternion (w,x,y,z) that rotates +Z → (x,y,z),
    and returns a new DataFrame with added columns: qw, qx, qy, qz.
    """
    # apply row‑wise
    qs = df.apply(
        lambda row: quaternion_from_direction((row[x_col], row[y_col], row[z_col])),
        axis=1,
        result_type='expand'
    )
    qs.columns = ['qx', 'qy', 'qz', 'qw']
    return df.join(qs)

def mads_type_to_unified_type(mads_type: str) -> AgentType:
    if mads_type.startswith("person"):
        return AgentType.PEDESTRIAN
    elif mads_type == "automobile":
        return AgentType.VEHICLE
    elif mads_type == "other_vehicle":
        return AgentType.VEHICLE
    elif mads_type.startswith("cycle"):
        return AgentType.BICYCLE
    elif mads_type.startswith("motorcycle"):
        return AgentType.MOTORCYCLE
    # v1 type
    elif "VEHICLE" in mads_type:
        return AgentType.VEHICLE
    elif "PEDESTRIAN" in mads_type:
        return AgentType.PEDESTRIAN
    elif "BIKE_MOTOR" in mads_type:
        return AgentType.MOTORCYCLE
    elif "BIKE" in mads_type:
        return AgentType.BICYCLE
    else:
        return AgentType.UNKNOWN


def _preprocess_lanes(
    positions: np.ndarray, lane_ids: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Build lane-id groups by relative position label.

    Args:
        positions: Lane position labels from `dw_lane.position`.
        lane_ids: Lane IDs aligned with `positions`.

    Returns:
        Dictionary with `ego` and optional `left`/`right` lane-id arrays.
    """
    assert len(positions) == len(
        lane_ids
    ), "positions and lane_ids must have the same length"

    lane_info: Dict[str, np.ndarray] = {}

    # From av/avprotos/lane.proto `LanePosition`.
    left_indices = np.where(positions == 2)[0]
    right_indices = np.where(positions == 3)[0]
    ego_indices = np.where(positions == 1)[0]

    lane_info["ego"] = lane_ids[ego_indices]

    if len(left_indices) > 0:
        lane_info["left"] = lane_ids[left_indices]
    if len(right_indices) > 0:
        lane_info["right"] = lane_ids[right_indices]

    return lane_info


def _process_single_lane(
    lane_data: Dict[str, Any],
    lane_chunks_details: pd.DataFrame,
    lane_info: Dict[str, np.ndarray],
    lane_ids_in_order: Sequence[int],
    rotation: np.ndarray,
    translation: np.ndarray,
) -> Tuple[int, Dict[str, Any]]:
    """Process one lane row into the intermediate lane map structure.

    Args:
        lane_data: Per-lane fields extracted from `dw_lane`.
        lane_chunks_details: Lane-chunk details table.
        lane_info: Lane groups with keys `ego`, and optionally `left`/`right`.
        lane_ids_in_order: Lane IDs aligned to continuation indices.
        rotation: 3x3 rotation matrix.
        translation: 3D translation vector.

    Returns:
        Tuple `(original_index, processed_lane_properties)`.
    """
    idx = lane_data["idx"]

    live_path_map_lane: Dict[str, Any] = {
        "beginType": int(lane_data["begin_type"]),
        "conf": 1.25,
        "endType": int(lane_data["end_type"]),
        "id": int(lane_data["current_id"]),
        "isBiDir": bool(lane_data["bidirectional"]),
        "isTurn": bool(lane_data["turns_allowed"]),
        "laneClass": int(lane_data["lane_class"]),
        "laneClassConf": 1.0,
        "ts": int(lane_data["timestamp"]),
    }

    lane_id = int(lane_data["current_id"])
    if lane_id in lane_info.get("ego", np.array([], dtype=np.int64)):
        right_lane_id = lane_info.get("right")
        left_lane_id = lane_info.get("left")
        if right_lane_id is not None:
            live_path_map_lane["laneChangeRightIds"] = right_lane_id.tolist()
        if left_lane_id is not None:
            live_path_map_lane["laneChangeLeftIds"] = left_lane_id.tolist()

    continuation_array = lane_data["continuation_array"]
    if len(continuation_array) > 0:
        live_path_map_lane["laneSuccessorIds"] = [
            lane_ids_in_order[idx_tmp] for idx_tmp in continuation_array
        ]

    chunk_indices = lane_data["chunk_indices"]
    live_path_map_lane["laneGeometry"] = _extract_lane_geometry(
        lane_chunks_details, chunk_indices, rotation, translation,
    )

    # Preserve original order via index for caller-side sorting.
    return (idx, live_path_map_lane)


def _extract_lane_geometry(
    lane_chunks_details: pd.DataFrame,
    chunk_indices: Sequence[int],
    rotation: np.ndarray,
    translation: np.ndarray,
) -> List[Dict[str, Any]]:
    """Extract and transform lane geometry samples for one lane.

    Args:
        lane_chunks_details: Lane chunk table for one timestamp.
        chunk_indices: Indices of chunk rows to use.
        rotation: 3x3 rotation matrix.
        translation: 3D translation vector.

    Returns:
        List of geometry dictionaries per sampled point.
    """
    filtered_lane_chunks = lane_chunks_details.iloc[list(chunk_indices)]
    # Check if df_expand_json has been called (clipgt-2.0.0).
    chunk_key = "lane_chunk"
    if "LaneChunk.center" in filtered_lane_chunks:
        chunk_key = "LaneChunk"

    # Keep a best-effort reference length for robust fallback arrays.
    # This preserves prior behavior intent (matching other per-point arrays)
    # while avoiding undefined-variable fallback paths.
    reference_len = 0

    def flatten_column(name: str) -> np.ndarray:
        """Flatten a ListArray column and convert to float32."""
        nonlocal reference_len
        try:
            # If expanded columns are present, use them directly.
            if name in filtered_lane_chunks:
                col = filtered_lane_chunks[name].to_list()
            else:
                # clipgt-2.0.0 fallback: each cell can be a list of json objects.
                col = []
                splitname = name.rsplit(".", 1)
                for row in filtered_lane_chunks[splitname[0]]:
                    col.append(pd.json_normalize(row)[splitname[1]])
            flat = np.concatenate(col).astype(np.float32)
            reference_len = max(reference_len, int(flat.shape[0]))
            return flat
        except (KeyError, ValueError):
            # Keep historical fallback behavior intent with a safe reference length.
            return np.zeros(reference_len, dtype=np.float32)

    flat_cx = flatten_column(f"{chunk_key}.center.x")
    flat_cy = flatten_column(f"{chunk_key}.center.y")
    flat_cz = flatten_column(f"{chunk_key}.center.z")

    flat_lx = flatten_column(f"{chunk_key}.left.x")
    flat_ly = flatten_column(f"{chunk_key}.left.y")
    flat_lz = flatten_column(f"{chunk_key}.left.z")

    flat_rx = flatten_column(f"{chunk_key}.right.x")
    flat_ry = flatten_column(f"{chunk_key}.right.y")
    flat_rz = flatten_column(f"{chunk_key}.right.z")

    centers = np.stack([flat_cx, flat_cy, flat_cz], axis=1)
    lefts = np.stack([flat_lx, flat_ly, flat_lz], axis=1)
    rights = np.stack([flat_rx, flat_ry, flat_rz], axis=1)

    centers = centers @ rotation.T + translation
    lefts = lefts @ rotation.T + translation
    rights = rights @ rotation.T + translation

    def flatten_int(name: str) -> np.ndarray:
        """Flatten a ListArray column and convert to int32."""
        try:
            col = filtered_lane_chunks[name].to_list()
            flat = np.concatenate(col).astype(np.int32)
            return flat
        except (KeyError, ValueError):
            return np.zeros(len(centers), dtype=np.int32)

    left_color = flatten_int(f"{chunk_key}.leftColor")
    left_style = flatten_int(f"{chunk_key}.leftStyle")
    left_type = flatten_int(f"{chunk_key}.leftType")
    right_color = flatten_int(f"{chunk_key}.rightColor")
    right_style = flatten_int(f"{chunk_key}.rightStyle")
    right_type = flatten_int(f"{chunk_key}.rightType")

    lane_geometry = [
        {
            "centerNormal": [0.0, 0.0, 0.0],
            "centerXYZ": centers[i].tolist(),
            "comSpeed": 0.0,
            "leftColor": int(left_color[i]),
            "leftStyle": int(left_style[i]),
            "leftType": int(left_type[i]),
            "leftXYZ": lefts[i].tolist(),
            "maxSpeed": 0.0,
            "rightColor": int(right_color[i]),
            "rightStyle": int(right_style[i]),
            "rightType": int(right_type[i]),
            "rightXYZ": rights[i].tolist(),
        }
        for i in range(len(centers))
    ]
    return lane_geometry


def _process_lanes_parallel(
    lane_table: pd.DataFrame,
    lane_patch_table: pd.DataFrame,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> List[Any]:
    """Process all lanes for one timestamp, optionally in parallel.

    Args:
        lane_table: `dw_lane` rows for one timestamp.
        lane_patch_table: `lane_chunk` rows for one timestamp.
        rotation: 3x3 rotation matrix.
        translation: 3D translation vector.

    Returns:
        List of per-lane dictionaries sorted by original lane index.
    """
    num_lanes = len(lane_table)

    # Check if df_expand_json has been called (clipgt-2.0.0).
    lane_key = "dw_lane"
    if "dw_lane.laneClass" not in lane_table:
        lane_key = "DwLane"

    timestamp_col = lane_table["key.timestamp_micros"].to_numpy()
    lane_class_col = lane_table[f"{lane_key}.laneClass"].to_numpy()
    begin_type_col = lane_table[f"{lane_key}.beginType"].to_numpy()
    end_type_col = lane_table[f"{lane_key}.endType"].to_numpy()
    current_id_col = lane_table[f"{lane_key}.currentId"].to_numpy()
    position_col = lane_table[f"{lane_key}.position"].to_numpy()
    bidirectional_col = lane_table[f"{lane_key}.bidirectional"].to_numpy()
    turns_allowed_col = lane_table[f"{lane_key}.turnsAllowed"].to_numpy()
    continuation_array = lane_table[
        f"{lane_key}.lgwm_lane_continuation_array"
    ].to_numpy()
    chunk_indices = lane_table[f"{lane_key}.chunkIndices"].to_numpy()

    lane_info = _preprocess_lanes(position_col, current_id_col)

    lane_data: List[Dict[str, Any]] = [
        {
            "idx": i,
            "timestamp": timestamp_col[i],
            "lane_class": lane_class_col[i],
            "begin_type": begin_type_col[i],
            "end_type": end_type_col[i],
            "current_id": current_id_col[i],
            "bidirectional": bidirectional_col[i],
            "turns_allowed": turns_allowed_col[i],
            "continuation_array": continuation_array[i],
            "chunk_indices": chunk_indices[i],
        }
        for i in range(num_lanes)
    ]

    process_func = partial(
        _process_single_lane,
        lane_chunks_details=lane_patch_table,
        lane_info=lane_info,
        lane_ids_in_order=current_id_col,
        rotation=rotation,
        translation=translation,
    )

    num_processes = min(multiprocessing.cpu_count(), num_lanes)

    # Process in parallel only when there are enough lanes to amortize overhead.
    if num_lanes > 4:
        with futures.ThreadPoolExecutor(max_workers=num_processes) as executor:
            results = list(executor.map(process_func, lane_data))
    else:
        # For small datasets, process sequentially to avoid thread overhead.
        results = [process_func(lane_item) for lane_item in lane_data]

    sorted_results: List[Any] = [None] * num_lanes
    for idx, result in results:
        sorted_results[int(idx)] = result

    return sorted_results


def _validate_map_timestamp_sampling(ts_list: np.ndarray, map_root: str) -> None:
    """Validate timestamp cadence and optionally emit rich debug artifacts."""
    expected_raw_delta_micros = 100_000
    sample_rate = 20
    expected_sampled_delta_micros = expected_raw_delta_micros * sample_rate

    raw_tolerance_micros = 500
    sampled_tolerance_micros = 10_000
    hard_gap_factor = 1.5

    raw_deltas = np.diff(ts_list)
    sampled_ts = ts_list[::sample_rate]
    sampled_deltas = np.diff(sampled_ts)

    raw_abs_err = np.abs(raw_deltas - expected_raw_delta_micros)
    sampled_abs_err = np.abs(sampled_deltas - expected_sampled_delta_micros)

    raw_within_tolerance = bool(np.all(raw_abs_err <= raw_tolerance_micros))
    sampled_within_tolerance = bool(
        np.all(sampled_abs_err <= sampled_tolerance_micros)
    )

    raw_hard_gap_threshold = int(expected_raw_delta_micros * hard_gap_factor)
    sampled_hard_gap_threshold = int(expected_sampled_delta_micros * hard_gap_factor)
    raw_hard_gap_mask = raw_deltas > raw_hard_gap_threshold
    sampled_hard_gap_mask = sampled_deltas > sampled_hard_gap_threshold
    has_meaningful_gaps = bool(np.any(raw_hard_gap_mask) or np.any(sampled_hard_gap_mask))

    debug_dir_override = os.getenv("MADS_TS_DEBUG_DIR")
    debug_dir_override_str = debug_dir_override.strip() if debug_dir_override else ""

    # Silent by default: only emit timestamp-gap warnings/debug artifacts when
    # explicit debug mode is enabled by providing MADS_TS_DEBUG_DIR.
    if not debug_dir_override_str:
        return

    force_debug = True
    base_debug_dir = debug_dir_override_str
    clip_tag = os.path.basename(os.path.normpath(map_root)) or "unknown_clip"
    clip_tag = clip_tag.replace(" ", "_")
    debug_dir = os.path.join(base_debug_dir, clip_tag)

    should_write_debug = has_meaningful_gaps
    if not should_write_debug:
        return

    try:
        os.makedirs(debug_dir, exist_ok=True)

        debug_csv_path = os.path.join(debug_dir, "mads_ts_debug.csv")
        summary_json_path = os.path.join(debug_dir, "mads_ts_debug_summary.json")

        is_sampled = np.zeros(ts_list.size, dtype=bool)
        is_sampled[::sample_rate] = True

        delta_to_next: List[Optional[int]] = [int(delta) for delta in raw_deltas]
        delta_to_next.append(None)

        nominal_step_to_next: List[Optional[bool]] = [
            bool(abs(int(delta) - expected_raw_delta_micros) <= raw_tolerance_micros)
            for delta in raw_deltas
        ]
        nominal_step_to_next.append(None)

        debug_df = pd.DataFrame(
            {
                "idx": np.arange(ts_list.size, dtype=np.int64),
                "timestamp_micros": ts_list.astype(np.int64),
                "delta_to_next_micros": delta_to_next,
                "is_sampled": is_sampled,
                "is_nominal_step_to_next": nominal_step_to_next,
            }
        )
        debug_df.to_csv(debug_csv_path, index=False)

        summary: Dict[str, Any] = {
            "num_timestamps": int(ts_list.size),
            "expected_raw_delta_micros": int(expected_raw_delta_micros),
            "expected_sampled_delta_micros": int(expected_sampled_delta_micros),
            "sample_rate_steps": int(sample_rate),
            "raw_tolerance_micros": int(raw_tolerance_micros),
            "sampled_tolerance_micros": int(sampled_tolerance_micros),
            "raw_within_tolerance": raw_within_tolerance,
            "sampled_within_tolerance": sampled_within_tolerance,
            "force_debug": force_debug,
            "num_raw_out_of_tolerance": int(np.sum(raw_abs_err > raw_tolerance_micros)),
            "num_sampled_out_of_tolerance": int(
                np.sum(sampled_abs_err > sampled_tolerance_micros)
            ),
            "raw_hard_gap_threshold_micros": int(raw_hard_gap_threshold),
            "sampled_hard_gap_threshold_micros": int(sampled_hard_gap_threshold),
            "num_raw_hard_gaps": int(np.sum(raw_hard_gap_mask)),
            "num_sampled_hard_gaps": int(np.sum(sampled_hard_gap_mask)),
            "has_meaningful_gaps": has_meaningful_gaps,
            "raw_delta_min_micros": (
                int(raw_deltas.min()) if raw_deltas.size > 0 else None
            ),
            "raw_delta_max_micros": (
                int(raw_deltas.max()) if raw_deltas.size > 0 else None
            ),
            "sampled_delta_min_micros": (
                int(sampled_deltas.min()) if sampled_deltas.size > 0 else None
            ),
            "sampled_delta_max_micros": (
                int(sampled_deltas.max()) if sampled_deltas.size > 0 else None
            ),
        }

        with open(summary_json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)

        if has_meaningful_gaps:
            warnings.warn(
                "Detected meaningful map timestamp gaps; sampling remains unchanged "
                f"(every {sample_rate}th entry). Debug files saved under: "
                f"{debug_dir}",
                stacklevel=2,
            )
    except OSError as exc:
        warnings.warn(
            "Timestamp validation found issues, but failed to write debug files "
            f"to {debug_dir}: {exc}",
            stacklevel=2,
        )

def _select_strict_2s_timestamps(
    ts_list: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Select timestamps on a strict 2s time grid with nearest-neighbor snapping."""
    grid_step_micros = 2_000_000
    base_dt_micros = 100_000
    sample_rate = 20
    max_snap_micros = int(os.getenv("MADS_MAP_STRICT_MAX_SNAP_US", "150000"))

    # 1) Fast exit for empty clips.
    if ts_list.size == 0:
        empty = np.array([], dtype=np.int64)
        stats: Dict[str, Any] = {
            "grid_step_micros": grid_step_micros,
            "max_snap_micros": max_snap_micros,
            "target_count": 0,
            "selected_count": 0,
            "coverage": 0.0,
            "max_abs_snap_error_micros": None,
            "sample_rate_steps": sample_rate,
        }
        return empty, empty, stats

    # 2) Build ideal 2s targets from the first to last timestamp.
    start_ts = int(ts_list[0])
    end_ts = int(ts_list[-1])
    targets = np.arange(start_ts, end_ts + 1, grid_step_micros, dtype=np.int64)

    right_idx = np.searchsorted(ts_list, targets, side="left")
    # 3) For each target, get nearest left/right candidates in ts_list.
    left_idx = np.clip(right_idx - 1, 0, ts_list.size - 1)
    right_idx_clipped = np.clip(right_idx, 0, ts_list.size - 1)

    left_ts = ts_list[left_idx]
    right_ts = ts_list[right_idx_clipped]
    # 4) Snap each target to whichever candidate is closer.
    choose_right = (right_idx < ts_list.size) & (
        np.abs(right_ts - targets) < np.abs(left_ts - targets)
    )
    chosen_idx = np.where(choose_right, right_idx_clipped, left_idx)

    # 5) Keep only snapped timestamps within max_snap_micros tolerance.
    chosen_ts = ts_list[chosen_idx]
    abs_snap_err = np.abs(chosen_ts - targets)
    keep_mask = abs_snap_err <= max_snap_micros

    kept_targets = targets[keep_mask]
    kept_ts = chosen_ts[keep_mask]

    # 6) Deduplicate in case nearby targets snap to the same source timestamp.
    if kept_ts.size > 0:
        unique_mask = np.concatenate(([True], kept_ts[1:] != kept_ts[:-1]))
        kept_targets = kept_targets[unique_mask]
        kept_ts = kept_ts[unique_mask]

    # 7) Convert kept 2s targets into trajdata frame units (0.1s per frame).
    #    Also compute summary stats for monitoring coverage/snap error.
    selected_frame_idx = ((kept_targets - start_ts) // base_dt_micros).astype(np.int64)
    max_abs_err = int(abs_snap_err[keep_mask].max()) if np.any(keep_mask) else None
    coverage = float(np.sum(keep_mask) / targets.size) if targets.size > 0 else 0.0

    stats = {
        "grid_step_micros": grid_step_micros,
        "max_snap_micros": max_snap_micros,
        "target_count": int(targets.size),
        "selected_count": int(kept_ts.size),
        "coverage": coverage,
        "max_abs_snap_error_micros": max_abs_err,
        "sample_rate_steps": sample_rate,
    }
    return kept_ts.astype(np.int64), selected_frame_idx, stats

def populate_vector_map(vector_map: VectorMap, map_root: str) -> None:
    """Populate `vector_map` from MADS lane parquet files.

    Args:
        vector_map: Target trajdata vector map to populate in-place.
        map_root: Directory containing `dw_lane.parquet` and `lane_chunk.parquet`.

    Returns:
        None.
    """
    maximum_bound: np.ndarray = np.full((3,), np.nan)
    minimum_bound: np.ndarray = np.full((3,), np.nan)

    rotation = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
    )
    translation = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    dw_lane_path = os.path.join(map_root, "dw_lane.parquet")
    lane_chunk_path = os.path.join(map_root, "lane_chunk.parquet")
    if not (os.path.exists(dw_lane_path) and os.path.exists(lane_chunk_path)):
        raise FileNotFoundError(
            f"Missing lane parquet(s) under {map_root}: "
            f"dw_lane.parquet={os.path.exists(dw_lane_path)}, "
            f"lane_chunk.parquet={os.path.exists(lane_chunk_path)}"
        )

    df_dw_lane = pd.read_parquet(dw_lane_path)
    df_lane_chunk = pd.read_parquet(lane_chunk_path)

    # Check if df_expand_json needs to be called (clipgt-2.0.0).
    if "key.timestamp_micros" not in df_dw_lane:
        df_dw_lane = df_expand_json(df_dw_lane)

    if "key.timestamp_micros" not in df_dw_lane:
        ts_list: np.ndarray = np.array([], dtype=np.int64)
    else:
        # Keep timestamp handling explicit to avoid Series/ndarray union issues in type stubs.
        ts_numeric = pd.Series(
            pd.to_numeric(df_dw_lane["key.timestamp_micros"], errors="coerce")
        )
        ts_valid = ts_numeric.loc[ts_numeric.notna()].to_numpy()
        ts_list = np.unique(np.asarray(ts_valid, dtype=np.int64))

    if "key.timestamp_micros" not in df_lane_chunk:
        df_lane_chunk = df_expand_json(df_lane_chunk)

    _validate_map_timestamp_sampling(ts_list=ts_list, map_root=map_root)

    selected_ts, selected_frame_idx, strict_stats = _select_strict_2s_timestamps(
        ts_list=ts_list
    )
    if strict_stats["target_count"] > 0 and strict_stats["selected_count"] == 0:
        raise ValueError(
            "Strict 2s timestamp sampling found no valid timestamps. "
            f"map_root={map_root}, stats={strict_stats}"
        )

    if strict_stats["target_count"] > 0 and strict_stats["coverage"] < 0.9:
        warnings.warn(
            "Low strict 2s timestamp coverage; map output may be sparse. "
            f"map_root={map_root}, stats={strict_stats}",
            stacklevel=2,
        )

    live_path_map_lanes: List[Tuple[int, List[Any]]] = []

    for frame_idx, ts in zip(selected_frame_idx, selected_ts):
        dw_mask = df_dw_lane["key.timestamp_micros"] == ts
        chunk_mask = df_lane_chunk["key.timestamp_micros"] == ts
        df_dw_lane_ts: pd.DataFrame = df_dw_lane.loc[dw_mask]
        df_lane_chunk_ts: pd.DataFrame = df_lane_chunk.loc[chunk_mask]

        live_path_map_lanes.append(
            (
                int(frame_idx),
                _process_lanes_parallel(
                    lane_table=df_dw_lane_ts,
                    lane_patch_table=df_lane_chunk_ts,
                    rotation=rotation,
                    translation=translation,
                ),
            )
        )  # (ts, map results)

    all_lanes_dict: Dict[str, Dict[str, Any]] = {}
    for ts, lanes_ts in live_path_map_lanes:  # per ts
        for lane in lanes_ts:
            lane_id = f"{lane['id']}_{ts}"
            lane_geometry = lane["laneGeometry"]

            left_rail: List[Any] = []
            right_rail: List[Any] = []
            midlane_pts: List[Any] = []
            next_lane: List[str] = []
            prev_lane: List[str] = []
            left_lane: List[str] = []
            right_lane: List[str] = []
            traffic_sign: List[Any] = []
            wait_line: List[Any] = []

            for segment in lane_geometry:
                left_rail.append(segment["leftXYZ"])
                right_rail.append(segment["rightXYZ"])
                midlane_pts.append(segment["centerXYZ"])

            left_rail_arr = np.array(left_rail).reshape(-1, 3)
            right_rail_arr = np.array(right_rail).reshape(-1, 3)
            midlane_pts_arr = np.array(midlane_pts).reshape(-1, 3)

            if "laneSuccessorIds" in lane:
                for lane_successor_id in lane["laneSuccessorIds"]:
                    next_lane.append(f"{lane_successor_id}_{ts}")

            if "lanePredecessorIds" in lane:
                for lane_predecessor_id in lane["lanePredecessorIds"]:
                    prev_lane.append(f"{lane_predecessor_id}_{ts}")

            if "laneChangeLeftIds" in lane:
                for lane_change_left_id in lane["laneChangeLeftIds"]:
                    left_lane.append(f"{lane_change_left_id}_{ts}")

            if "laneChangeRightIds" in lane:
                for lane_change_right_id in lane["laneChangeRightIds"]:
                    right_lane.append(f"{lane_change_right_id}_{ts}")

            all_lanes_dict[lane_id] = {
                "left_rail": left_rail_arr,
                "right_rail": right_rail_arr,
                "midlane_pts": midlane_pts_arr,
                "next_lane": next_lane,
                "prev_lane": prev_lane,
                "left_lane": left_lane,
                "right_lane": right_lane,
                "traffic_sign": traffic_sign,
                "wait_line": wait_line,
            }

    if not all_lanes_dict:
        print("No valid data available in map file")
        return

    # Creating Vectorized Map.
    for lane_id, lane_info_dict in all_lanes_dict.items():
        left_polyline = np.asarray(lane_info_dict["left_rail"], dtype=np.float32)
        right_polyline = np.asarray(lane_info_dict["right_rail"], dtype=np.float32)
        midlane_pts = np.asarray(lane_info_dict["midlane_pts"], dtype=np.float32)

        # Compute map bounds.
        left_max = np.max(left_polyline, axis=0)
        left_min = np.min(left_polyline, axis=0)
        maximum_bound = np.fmax(maximum_bound, left_max)
        minimum_bound = np.fmin(minimum_bound, left_min)

        right_max = np.max(right_polyline, axis=0)
        right_min = np.min(right_polyline, axis=0)
        maximum_bound = np.fmax(maximum_bound, right_max)
        minimum_bound = np.fmin(minimum_bound, right_min)

        mid_max = np.max(midlane_pts, axis=0)
        mid_min = np.min(midlane_pts, axis=0)
        maximum_bound = np.fmax(maximum_bound, mid_max)
        minimum_bound = np.fmin(minimum_bound, mid_min)

        new_lane = RoadLane(
            id=lane_id,
            center=Polyline(midlane_pts).interpolate(max_dist=MAX_POLYLINE_POINT_DIST),
            left_edge=Polyline(left_polyline).interpolate(
                max_dist=MAX_POLYLINE_POINT_DIST
            ),
            right_edge=Polyline(right_polyline).interpolate(
                max_dist=MAX_POLYLINE_POINT_DIST
            ),
            next_lanes=lane_info_dict["next_lane"],
            adj_lanes_left=lane_info_dict["left_lane"],
            adj_lanes_right=lane_info_dict["right_lane"],
            prev_lanes=lane_info_dict["prev_lane"],
        )
        vector_map.add_map_element(new_lane)

    # vector_map.extent is [min_x, min_y, min_z, max_x, max_y, max_z]
    vector_map.extent = np.concatenate((minimum_bound, maximum_bound))
