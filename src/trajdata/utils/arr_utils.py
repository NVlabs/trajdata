from enum import IntEnum
from typing import List, Optional

import numpy as np
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence


class PadDirection(IntEnum):
    BEFORE = 0
    AFTER = 1


def convert_with_dir(
    seq: np.ndarray, dtype: torch.dtype, time_dim: int, pad_dir: PadDirection
) -> Tensor:
    if pad_dir == PadDirection.BEFORE:
        return torch.as_tensor(seq, dtype=dtype).flip(time_dim)

    return torch.as_tensor(seq, dtype=dtype)


def pad_with_dir(
    seq_list: List[Tensor], time_dim: int, pad_dir: PadDirection, **kwargs
) -> Tensor:
    if pad_dir == PadDirection.BEFORE:
        return pad_sequence(
            seq_list,
            **kwargs,
        ).flip(time_dim)

    return pad_sequence(
        seq_list,
        **kwargs,
    )


def pad_sequences(
    seq_list: List[np.ndarray],
    dtype: torch.dtype,
    time_dim: int,
    pad_dir: PadDirection,
    **kwargs,
) -> Tensor:
    return pad_with_dir(
        [convert_with_dir(seq, dtype, time_dim, pad_dir) for seq in seq_list],
        time_dim,
        pad_dir,
        **kwargs,
    )


def mask_up_to(lens: Tensor, delta: int = 0, max_len: Optional[int] = None) -> Tensor:
    """Exclusive.

    Args:
        lens (Tensor): _description_
        delta (int, optional): _description_. Defaults to 0.

    Returns:
        Tensor: _description_
    """
    if max_len is None:
        max_len = lens.max()

    arange_t: Tensor = torch.arange(
        max_len, dtype=lens.dtype, device=lens.device
    ).expand(*lens.shape, -1)

    return arange_t < (lens.unsqueeze(-1) + delta)


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


def batch_nd_transform_points_np(points: np.ndarray, Mat: np.ndarray) -> np.ndarray:
    ndim = Mat.shape[-1] - 1
    batch = list(range(Mat.ndim - 2)) + [Mat.ndim - 1] + [Mat.ndim - 2]
    Mat = np.transpose(Mat, batch)
    if points.ndim == Mat.ndim - 1:
        return (points[..., np.newaxis, :] @ Mat[..., :ndim, :ndim]).squeeze(-2) + Mat[
            ..., -1:, :ndim
        ].squeeze(-2)
    elif points.ndim == Mat.ndim:
        return (
            (points[..., np.newaxis, :] @ Mat[..., np.newaxis, :ndim, :ndim])
            + Mat[..., np.newaxis, -1:, :ndim]
        ).squeeze(-2)
    else:
        raise Exception("wrong shape")


def batch_nd_transform_angles_np(angles: np.ndarray, Mat: np.ndarray) -> np.ndarray:
    cos_vals, sin_vals = Mat[..., 0, 0], Mat[..., 1, 0]
    rot_angle = np.arctan2(sin_vals, cos_vals)
    angles = angles + rot_angle
    angles = angle_wrap(angles)
    return angles


def batch_nd_transform_points_angles_np(
    points_angles: np.ndarray, Mat: np.ndarray
) -> np.ndarray:
    assert points_angles.shape[-1] == 3
    points = batch_nd_transform_points_np(points_angles[..., :2], Mat)
    angles = batch_nd_transform_angles_np(points_angles[..., 2:3], Mat)
    points_angles = np.concatenate([points, angles], axis=-1)
    return points_angles


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


def batch_proj(x, line):
    # x:[batch,3], line:[batch,N,3]
    line_length = line.shape[-2]
    batch_dim = x.ndim - 1
    if isinstance(x, torch.Tensor):
        delta = line[..., 0:2] - torch.unsqueeze(x[..., 0:2], dim=-2).repeat(
            *([1] * batch_dim), line_length, 1
        )
        dis = torch.linalg.norm(delta, axis=-1)
        idx0 = torch.argmin(dis, dim=-1)
        idx = idx0.view(*line.shape[:-2], 1, 1).repeat(
            *([1] * (batch_dim + 1)), line.shape[-1]
        )
        line_min = torch.squeeze(torch.gather(line, -2, idx), dim=-2)
        dx = x[..., None, 0] - line[..., 0]
        dy = x[..., None, 1] - line[..., 1]
        delta_y = -dx * torch.sin(line_min[..., None, 2]) + dy * torch.cos(
            line_min[..., None, 2]
        )
        delta_x = dx * torch.cos(line_min[..., None, 2]) + dy * torch.sin(
            line_min[..., None, 2]
        )

        delta_psi = angle_wrap(x[..., 2] - line_min[..., 2])

        return (
            delta_x,
            delta_y,
            torch.unsqueeze(delta_psi, dim=-1),
        )

    elif isinstance(x, np.ndarray):
        delta = line[..., 0:2] - np.repeat(
            x[..., np.newaxis, 0:2], line_length, axis=-2
        )
        dis = np.linalg.norm(delta, axis=-1)
        idx0 = np.argmin(dis, axis=-1)
        idx = idx0.reshape(*line.shape[:-2], 1, 1).repeat(line.shape[-1], axis=-1)
        line_min = np.squeeze(np.take_along_axis(line, idx, axis=-2), axis=-2)
        dx = x[..., None, 0] - line[..., 0]
        dy = x[..., None, 1] - line[..., 1]
        delta_y = -dx * np.sin(line_min[..., None, 2]) + dy * np.cos(
            line_min[..., None, 2]
        )
        delta_x = dx * np.cos(line_min[..., None, 2]) + dy * np.sin(
            line_min[..., None, 2]
        )
        delta_psi = angle_wrap(x[..., 2] - line_min[..., 2])
        return (
            delta_x,
            delta_y,
            np.expand_dims(delta_psi, axis=-1),
        )


def quaternion_to_yaw(q: np.ndarray):
    # From https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L1025
    return np.arctan2(
        2 * (q[..., 0] * q[..., 3] - q[..., 1] * q[..., 2]),
        1 - 2 * (q[..., 2] ** 2 + q[..., 3] ** 2),
    )
