import unittest

import numpy as np
import torch

from trajdata.utils import arr_utils


class TestCollation(unittest.TestCase):
    def test_convert_with_dir(self):
        input = np.array([1, 2, 3])
        converted = arr_utils.convert_with_dir(
            input, torch.long, -1, arr_utils.PadDirection.AFTER
        )

        self.assertTrue(torch.allclose(torch.tensor(input), converted))

        input = np.array([1, 2, 3])
        converted = arr_utils.convert_with_dir(
            input, torch.long, -1, arr_utils.PadDirection.BEFORE
        )

        self.assertTrue(torch.allclose(torch.tensor(np.array([3, 2, 1])), converted))

    def test_pad_with_dir(self):
        inputs = [
            torch.tensor([[6], [5], [4], [3], [2], [1], [0]], dtype=torch.float),
            torch.tensor([[3], [2], [1], [0]], dtype=torch.float),
            torch.tensor([[9], [10], [11]], dtype=torch.float),
            torch.tensor([[9], [10], [11], [12], [13], [14]], dtype=torch.float),
        ]
        converted = arr_utils.pad_with_dir(
            inputs,
            -2,
            arr_utils.PadDirection.AFTER,
            batch_first=True,
            padding_value=np.nan,
        )

        self.assertTrue(
            torch.allclose(
                torch.tensor(
                    [
                        [[6], [5], [4], [3], [2], [1], [0]],
                        [[3], [2], [1], [0], [np.nan], [np.nan], [np.nan]],
                        [[9], [10], [11], [np.nan], [np.nan], [np.nan], [np.nan]],
                        [[9], [10], [11], [12], [13], [14], [np.nan]],
                    ],
                ),
                converted,
                equal_nan=True,
            )
        )

        converted = arr_utils.pad_with_dir(
            inputs,
            -2,
            arr_utils.PadDirection.BEFORE,
            batch_first=True,
            padding_value=np.nan,
        )

        self.assertTrue(
            torch.allclose(
                torch.tensor(
                    [
                        [[0], [1], [2], [3], [4], [5], [6]],
                        [[np.nan], [np.nan], [np.nan], [0], [1], [2], [3]],
                        [[np.nan], [np.nan], [np.nan], [np.nan], [11], [10], [9]],
                        [[np.nan], [14], [13], [12], [11], [10], [9]],
                    ],
                ),
                converted,
                equal_nan=True,
            )
        )

    def test_pad_sequences(self):
        inputs = [
            np.array([[6], [5], [4], [3], [2], [1], [0]], dtype=float),
            np.array([[3], [2], [1], [0]], dtype=float),
            np.array([[9], [10], [11]], dtype=float),
            np.array([[9], [10], [11], [12], [13], [14]], dtype=float),
        ]
        converted = arr_utils.pad_sequences(
            inputs,
            torch.float,
            -2,
            arr_utils.PadDirection.AFTER,
            batch_first=True,
            padding_value=np.nan,
        )

        self.assertTrue(
            torch.allclose(
                torch.tensor(
                    [
                        [[6], [5], [4], [3], [2], [1], [0]],
                        [[3], [2], [1], [0], [np.nan], [np.nan], [np.nan]],
                        [[9], [10], [11], [np.nan], [np.nan], [np.nan], [np.nan]],
                        [[9], [10], [11], [12], [13], [14], [np.nan]],
                    ],
                ),
                converted,
                equal_nan=True,
            )
        )

        converted = arr_utils.pad_sequences(
            inputs,
            torch.float,
            -2,
            arr_utils.PadDirection.BEFORE,
            batch_first=True,
            padding_value=np.nan,
        )

        self.assertTrue(
            torch.allclose(
                torch.tensor(
                    [
                        [[6], [5], [4], [3], [2], [1], [0]],
                        [[np.nan], [np.nan], [np.nan], [3], [2], [1], [0]],
                        [[np.nan], [np.nan], [np.nan], [np.nan], [9], [10], [11]],
                        [[np.nan], [9], [10], [11], [12], [13], [14]],
                    ],
                ),
                converted,
                equal_nan=True,
            )
        )
