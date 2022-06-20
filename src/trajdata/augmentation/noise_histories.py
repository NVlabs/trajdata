import torch

from trajdata.augmentation.augmentation import BatchAugmentation
from trajdata.data_structures.batch import AgentBatch


class NoiseHistories(BatchAugmentation):
    def __init__(self, mean: float = 0.0, stddev: float = 0.1) -> None:
        self.mean = mean
        self.stddev = stddev

    def apply(self, agent_batch: AgentBatch) -> None:
        agent_batch.agent_hist[..., :-1, :] += torch.normal(
            self.mean, self.stddev, size=agent_batch.agent_hist[..., :-1, :].shape
        )
        agent_batch.neigh_hist[..., :-1, :] += torch.normal(
            self.mean, self.stddev, size=agent_batch.neigh_hist[..., :-1, :].shape
        )
