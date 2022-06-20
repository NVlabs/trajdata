import numpy as np
import torch
from torch import Tensor


def vrange(starts: np.ndarray, stops: np.ndarray) -> np.ndarray:
    """Create concatenated ranges of integers for multiple start/stop

    Parameters:
        starts (1-D array): starts for each range
        stops (1-D array): stops for each range (same shape as starts)

    Returns:
        numpy.ndarray: concatenated ranges

    For example:

        >>> starts = np.array([1, 3, 4, 6])
        >>> stops  = np.array([1, 5, 7, 6])
        >>> vrange(starts, stops)
        array([3, 4, 4, 5, 6])

    """
    lens = stops - starts
    return np.repeat(stops - lens.cumsum(), lens) + np.arange(lens.sum())


def angle_wrap(radians: np.ndarray) -> np.ndarray:
    """This function wraps angles to lie within [-pi, pi).

    Args:
        radians (np.ndarray): The input array of angles (in radians).

    Returns:
        np.ndarray: Wrapped angles that lie within [-pi, pi).
    """
    return (radians + np.pi) % (2 * np.pi) - np.pi


def rotation_matrix(angle: float) -> np.ndarray:
    """Creates a 2D rotation matrix.

    Args:
        angle (float): The angle to rotate points by.

    Returns:
        np.ndarray: The 2x2 rotation matrix.
    """
    return np.array(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ]
    )


def transform_matrices(angles: Tensor, translations: Tensor) -> Tensor:
    """Creates a 3x3 transformation matrix for each angle and translation in the input.

    Args:
        angles (Tensor): The (N,)-shaped angles tensor to rotate points by.
        translations (Tensor): The (N,2)-shaped translations to shift points by.

    Returns:
        Tensor: The Nx3x3 transformation matrices.
    """
    cos_vals = torch.cos(angles)
    sin_vals = torch.sin(angles)
    last_rows = torch.tensor(
        [[0.0, 0.0, 1.0]], dtype=angles.dtype, device=angles.device
    ).expand((angles.shape[0], -1))
    return torch.stack(
        [
            torch.stack([cos_vals, -sin_vals, translations[:, 0]], dim=-1),
            torch.stack([sin_vals, cos_vals, translations[:, 1]], dim=-1),
            last_rows,
        ],
        dim=-2,
    )


def agent_aware_diff(values: np.ndarray, agent_ids: np.ndarray) -> np.ndarray:
    values_diff: np.ndarray = np.diff(
        values, axis=0, prepend=values[[0]] - (values[[1]] - values[[0]])
    )

    # The point of the border mask is to catch data like this:
    # index    agent_id     vx    vy
    #     0           1    7.3   9.1
    #     1           2    0.0   0.0
    #                  ...
    # As implemented, we're not touching the very last row (we don't care anyways
    # for agents detected only once at a single timestep) and we would currently only
    # return index 0 (since we chop off the top with the 1: in the slice below), but
    # we want to return 1 so that's why the + 1 at the end.
    border_mask: np.ndarray = np.nonzero(agent_ids[1:-1] != agent_ids[:-2])[0] + 1
    values_diff[border_mask] = values_diff[border_mask + 1]

    return values_diff
