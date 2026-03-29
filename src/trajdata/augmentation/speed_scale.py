"""
SpeedScaleAugmentation: randomly scale agent speeds (and derived quantities)
at batch time to simulate faster or slower motion.

Scales: velocity (xd, yd), acceleration (xdd, ydd) and the temporal extent of
trajectories are left untouched – only the magnitude of movement is altered.

Usage::

    from trajdata.augmentation import SpeedScaleAugmentation
    dataset = UnifiedDataset(..., augmentations=[SpeedScaleAugmentation(0.7, 1.3)])
"""
import torch

from trajdata.augmentation.augmentation import BatchAugmentation
from trajdata.data_structures.batch import AgentBatch, SceneBatch


class SpeedScaleAugmentation(BatchAugmentation):
    """Randomly scale agent velocities and accelerations.

    A random scalar drawn uniformly from ``[scale_min, scale_max]`` is
    applied per sample to velocity channels (xd, yd → channels 3,4) and
    acceleration channels (xdd, ydd → channels 5,6).  Position channels are
    left unchanged so that the history/future trajectories remain geometrically
    consistent while the speed distribution is augmented.

    Args:
        scale_min: Lower bound of the uniform scale factor (default 0.8).
        scale_max: Upper bound of the uniform scale factor (default 1.2).
    """

    # Internal state format: x(0), y(1), z(2), xd(3), yd(4), xdd(5), ydd(6), h(7)
    _VEL_CHANNELS = (3, 4)
    _ACC_CHANNELS = (5, 6)

    def __init__(self, scale_min: float = 0.8, scale_max: float = 1.2) -> None:
        if scale_min <= 0 or scale_max <= 0:
            raise ValueError("Scale bounds must be positive.")
        if scale_min > scale_max:
            raise ValueError("scale_min must be ≤ scale_max.")
        self.scale_min = scale_min
        self.scale_max = scale_max

    # -----------------------------------------------------------------

    def _scale_traj(self, traj: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Scale velocity/acc channels; traj shape [B, T, D] or [B, N, T, D]."""
        result = traj.clone()
        D = traj.shape[-1]
        B = traj.shape[0]

        for ch in self._VEL_CHANNELS + self._ACC_CHANNELS:
            if ch < D:
                sliced = result[..., ch]  # [B, T] or [B, N, T]
                s = scale.view((B,) + (1,) * (sliced.dim() - 1))
                result[..., ch] = sliced * s.expand_as(sliced)

        return result

    def _sample_scale(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return (
            torch.rand(batch_size, device=device) * (self.scale_max - self.scale_min)
            + self.scale_min
        )

    # -----------------------------------------------------------------

    def apply_agent(self, batch: AgentBatch) -> None:
        B = batch.agent_hist.shape[0]
        scale = self._sample_scale(B, batch.agent_hist.device)

        batch.agent_hist = self._scale_traj(batch.agent_hist, scale)
        batch.agent_fut = self._scale_traj(batch.agent_fut, scale)

        if batch.neigh_hist is not None and batch.neigh_hist.numel() > 0:
            batch.neigh_hist = self._scale_traj(batch.neigh_hist, scale)
        if batch.neigh_fut is not None and batch.neigh_fut.numel() > 0:
            batch.neigh_fut = self._scale_traj(batch.neigh_fut, scale)

    def apply_scene(self, batch: SceneBatch) -> None:
        B = batch.agent_hist.shape[0]
        scale = self._sample_scale(B, batch.agent_hist.device)
        batch.agent_hist = self._scale_traj(batch.agent_hist, scale)
        batch.agent_fut = self._scale_traj(batch.agent_fut, scale)
