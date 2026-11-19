import unittest

import numpy as np
import pandas as pd

from trajdata.augmentation import LowSpeedYawCorrection


class TestLowSpeedYawCorrection(unittest.TestCase):
    def _scene_df(self) -> pd.DataFrame:
        # Two agents with four timesteps each. Agent "a" moves fast,
        # agent "b" is stopped and its heading jitters.
        index = pd.MultiIndex.from_product(
            [["a", "b"], range(4)], names=["agent_id", "scene_ts"]
        )
        return pd.DataFrame(
            {
                "vx": [5.0, 5.0, 5.0, 5.0, 0.0, 0.0, 0.0, 0.0],
                "vy": [0.0] * 8,
                "heading": [0.0, 0.1, 0.2, 0.3, 1.0, 1.5, 0.5, 2.0],
            },
            index=index,
        )

    def test_low_speed_heading_held(self):
        df = self._scene_df()
        LowSpeedYawCorrection(speed_threshold=1.0).apply(df)

        np.testing.assert_allclose(
            df["heading"].to_numpy(), [0.0, 0.1, 0.2, 0.3, 1.0, 1.0, 1.0, 1.0]
        )
        self.assertNotIn("yaw_diffs", df.columns)

    def test_zero_threshold_keeps_heading(self):
        df = self._scene_df()
        expected = df["heading"].to_numpy().copy()
        LowSpeedYawCorrection(speed_threshold=0.0).apply(df)

        np.testing.assert_allclose(df["heading"].to_numpy(), expected)


if __name__ == "__main__":
    unittest.main()
