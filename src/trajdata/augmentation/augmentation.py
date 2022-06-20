import pandas as pd

from trajdata.data_structures.batch import AgentBatch


class Augmentation:
    def __init__(self) -> None:
        raise NotImplementedError()


class DatasetAugmentation(Augmentation):
    def apply(self, scene_data_df: pd.DataFrame) -> None:
        raise NotImplementedError()


class BatchAugmentation(Augmentation):
    def apply(self, agent_batch: AgentBatch) -> None:
        raise NotImplementedError()
