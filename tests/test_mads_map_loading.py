import pandas as pd

from trajdata.dataset_specific.mads.mads_utils import _prepare_wait_lines


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
