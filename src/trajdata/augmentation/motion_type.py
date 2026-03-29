"""
MotionTypeLabeler: classify agent motion into discrete categories based on
instantaneous speed, and store the label as a batch extra.

Categories (stored as integer in ``batch.extras["motion_type"]``):

+----+--------------+---------------------------+
| ID | Name         | Speed range (m/s)         |
+====+==============+===========================+
|  0 | STATIONARY   | v < ``stationary_thresh`` |
|  1 | WALKING      | stationary – walking      |
|  2 | RUNNING      | walking – running         |
|  3 | FAST         | > ``running_thresh``      |
+----+--------------+---------------------------+

Usage::

    from trajdata.augmentation import MotionTypeLabeler
    dataset = UnifiedDataset(..., augmentations=[MotionTypeLabeler()])
    # batch.extras["motion_type"] → LongTensor [B]
"""
import torch

from trajdata.augmentation.augmentation import BatchAugmentation
from trajdata.data_structures.batch import AgentBatch, SceneBatch

STATIONARY = 0
WALKING = 1
RUNNING = 2
FAST = 3


class MotionTypeLabeler(BatchAugmentation):
    """Add a ``motion_type`` integer label to ``batch.extras``.

    The label is derived from the mean speed over the observed history
    (velocity channels xd, yd at indices 3 and 4 of the state).

    Args:
        stationary_thresh: Max speed to be considered stationary (default 0.5 m/s).
        walking_thresh: Max speed for walking category (default 2.5 m/s).
        running_thresh: Max speed for running category (default 6.0 m/s).
            Anything above is labelled FAST.
    """

    def __init__(
        self,
        stationary_thresh: float = 0.5,
        walking_thresh: float = 2.5,
        running_thresh: float = 6.0,
    ) -> None:
        self.stationary_thresh = stationary_thresh
        self.walking_thresh = walking_thresh
        self.running_thresh = running_thresh

    # -----------------------------------------------------------------

    def _compute_labels(self, hist: torch.Tensor, hist_len: torch.Tensor) -> torch.Tensor:
        """Compute per-sample motion type label from history tensor [B, T, D]."""
        D = hist.shape[-1]
        if D < 5:
            # Not enough channels to extract velocity
            return torch.zeros(hist.shape[0], dtype=torch.long, device=hist.device)

        vx = hist[..., 3]  # [B, T]
        vy = hist[..., 4]  # [B, T]
        speed = torch.sqrt(vx ** 2 + vy ** 2)  # [B, T]

        # Mean speed over valid timesteps only
        # hist_len: [B]
        T = hist.shape[1]
        valid_mask = (
            torch.arange(T, device=hist.device).unsqueeze(0) < hist_len.unsqueeze(1)
        ).float()  # [B, T]
        mean_speed = (speed * valid_mask).sum(dim=1) / hist_len.float().clamp(min=1)

        labels = torch.zeros(hist.shape[0], dtype=torch.long, device=hist.device)
        labels[mean_speed >= self.stationary_thresh] = WALKING
        labels[mean_speed >= self.walking_thresh] = RUNNING
        labels[mean_speed >= self.running_thresh] = FAST

        return labels

    # -----------------------------------------------------------------

    def apply_agent(self, batch: AgentBatch) -> None:
        labels = self._compute_labels(batch.agent_hist, batch.agent_hist_len)
        batch.extras["motion_type"] = labels

    def apply_scene(self, batch: SceneBatch) -> None:
        # For scene-centric batches, compute per-agent labels [B, A]
        B, A, T, D = batch.agent_hist.shape
        flat_hist = batch.agent_hist.view(B * A, T, D)
        flat_len = batch.agent_hist_len.view(B * A)
        labels = self._compute_labels(flat_hist, flat_len).view(B, A)
        batch.extras["motion_type"] = labels
