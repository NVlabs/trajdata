"""
MirrorAugmentation: randomly flip trajectories horizontally (x-axis) or
vertically (y-axis) at batch time.

Applies to: agent_hist, agent_fut, neigh_hist, neigh_fut.
Heading and velocity components are adjusted consistently.

Usage::

    from trajdata.augmentation import MirrorAugmentation
    dataset = UnifiedDataset(..., augmentations=[MirrorAugmentation(axis="x", prob=0.5)])
"""
import torch

from trajdata.augmentation.augmentation import BatchAugmentation
from trajdata.data_structures.batch import AgentBatch, SceneBatch


class MirrorAugmentation(BatchAugmentation):
    """Randomly mirror trajectories along the x or y axis.

    Args:
        axis: Which axis to mirror – ``"x"`` flips the x-coordinate and
              ``"y"`` flips the y-coordinate (default ``"x"``).
        prob: Probability of applying the flip to each sample (default 0.5).
    """

    def __init__(self, axis: str = "x", prob: float = 0.5) -> None:
        if axis not in ("x", "y"):
            raise ValueError("axis must be 'x' or 'y'")
        self.axis = axis
        self.prob = prob

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _flip_traj(self, traj: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Flip a [B, T, D] or [B, N, T, D] trajectory tensor.

        Internal state format: x(0) y(1) z(2) xd(3) yd(4) xdd(5) ydd(6) h(7).
        mask: bool tensor of shape [B] – True = apply flip.
        """
        result = traj.clone()
        D = traj.shape[-1]
        B = traj.shape[0]

        # Channels to negate
        if self.axis == "x":
            flip_chs = [c for c in (0, 3, 5, 7) if c < D]
        else:
            flip_chs = [c for c in (1, 4, 6, 7) if c < D]

        for ch in flip_chs:
            sliced = result[..., ch]          # shape [B, T] or [B, N, T]
            # Build mask with same ndim as sliced: [B, 1, ...1]
            m = mask.view((B,) + (1,) * (sliced.dim() - 1))
            result[..., ch] = torch.where(m.expand_as(sliced), -sliced, sliced)

        return result

    def _sample_mask(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.rand(batch_size, device=device) < self.prob

    # -----------------------------------------------------------------

    def apply_agent(self, batch: AgentBatch) -> None:
        B = batch.agent_hist.shape[0]
        mask = self._sample_mask(B, batch.agent_hist.device)

        batch.agent_hist = self._flip_traj(batch.agent_hist, mask)
        batch.agent_fut = self._flip_traj(batch.agent_fut, mask)

        if batch.neigh_hist is not None and batch.neigh_hist.numel() > 0:
            # neigh tensors are [B, N, T, D]; expand mask to [B]
            batch.neigh_hist = self._flip_traj(batch.neigh_hist, mask)
        if batch.neigh_fut is not None and batch.neigh_fut.numel() > 0:
            batch.neigh_fut = self._flip_traj(batch.neigh_fut, mask)

    def apply_scene(self, batch: SceneBatch) -> None:
        B = batch.agent_hist.shape[0]
        mask = self._sample_mask(B, batch.agent_hist.device)
        batch.agent_hist = self._flip_traj(batch.agent_hist, mask)
        batch.agent_fut = self._flip_traj(batch.agent_fut, mask)
