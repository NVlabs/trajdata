from uuid import UUID

import numpy as np
import pandas as pd
import pytest

from trajdata.dataset_specific.mads.mads_utils import (
    _prepare_wait_lines,
    find_lane_polylines_parquet,
)


def test_prepare_wait_lines_drops_rows_without_map_ids():
    wait_lines = pd.DataFrame(
        {
            "key.map_id": [None, "wait-42"],
            "WaitLine.location": [[], [{"x": 0.0, "y": 0.0, "z": 0.0}]],
        }
    )

    prepared = _prepare_wait_lines(wait_lines)

    assert prepared["key.map_id"].tolist() == ["wait-42"]
    assert prepared["lane.map_id"].tolist() == ["42"]
    assert prepared.index.tolist() == [0]


@pytest.mark.parametrize("nested_rails", [False, True])
@pytest.mark.parametrize("array_subjects", [False, True])
def test_lane_relations_preserve_clip_filter_first_match_and_wait_line_order(
    nested_rails,
    array_subjects,
):
    clip_id, other_clip = UUID(int=1), UUID(int=2)
    lane_a, lane_b, lane_c = (UUID(int=i) for i in range(3, 6))
    left = np.array([[0.0, 1.0, 0.0], [2.0, 1.0, 0.0]])
    right = left - np.array([0.0, 2.0, 0.0])
    lanes = {"key.map_id": [lane_a, lane_b, lane_c]}
    for side, points in (("left", left), ("right", right)):
        if nested_rails:
            lanes[f"Lane.{side}_rail"] = [
                [dict(zip("xyz", point)) for point in points]
            ] * 3
        else:
            for dimension, coordinates in zip("xyz", points.T):
                lanes[f"Lane.{side}_rail.{dimension}"] = [coordinates] * 3
    relations = pd.DataFrame(
        [
            [other_clip, lane_a, "NEXT_LANE", [lane_c]],
            [clip_id, lane_a, "NEXT_LANE", [lane_b]],
            [clip_id, lane_a, "NEXT_LANE", [lane_c]],
            [clip_id, lane_b, "PREVIOUS_LANE", [lane_a]],
            [clip_id, lane_a, "LEFT_LANE", []],
            [clip_id, lane_a, "LEFT_LANE", [lane_c]],
            [clip_id, lane_a, "RIGHT_LANE", [lane_c]],
            [clip_id, lane_a, "SIGN_TO_LANE", ["sign-a"]],
        ],
        columns=[
            "key.clip_id",
            "Association.subjects",
            "key.kind",
            "Association.objects",
        ],
    )
    if array_subjects:
        relations["Association.subjects"] = relations["Association.subjects"].map(
            lambda subject: np.array([subject])
        )
    wait_lines = pd.DataFrame(
        [
            [clip_id, lane_a, "wait-b"],
            [other_clip, lane_a, "other-wait"],
            [clip_id, lane_a, "wait-a"],
        ],
        columns=["key.clip_id", "lane.map_id", "key.map_id"],
    )

    result = find_lane_polylines_parquet(
        pd.DataFrame(lanes), relations, clip_id, wait_lines
    )

    assert list(result) == [lane_a, lane_b, lane_c]
    assert result[lane_a]["next_lane"] == [lane_b]
    assert result[lane_b]["prev_lane"] == [lane_a]
    assert result[lane_a]["left_lane"] == []
    assert result[lane_a]["right_lane"] == [lane_c]
    assert result[lane_a]["traffic_sign"] == ["sign-a"]
    assert result[lane_a]["wait_line"] == ["wait-b", "wait-a"]
    for kind in (
        "next_lane",
        "prev_lane",
        "left_lane",
        "right_lane",
        "traffic_sign",
        "wait_line",
    ):
        assert result[lane_c][kind] == {}
    for lane in result.values():
        np.testing.assert_array_equal(lane["left_rail"], left)
        np.testing.assert_array_equal(lane["right_rail"], right)


def test_empty_lanes_need_no_relation_columns():
    assert (
        find_lane_polylines_parquet(
            pd.DataFrame(), pd.DataFrame(), "clip", pd.DataFrame()
        )
        == {}
    )
