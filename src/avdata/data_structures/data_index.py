from typing import List, Tuple

import numpy as np


class DataIndex:
    """The data index is effectively a big list of tuples taking the form:

    (scene_path: str, scene_elem_index: int)
    """

    def __init__(self, index_elems: List[Tuple[str, int]]) -> None:
        scene_paths, scene_index_lens = zip(*index_elems)

        self.cumulative_lengths: np.ndarray = np.concatenate(
            ([0], np.cumsum(scene_index_lens))
        )
        self.len: int = self.cumulative_lengths[-1].item()

        self.scene_paths: np.ndarray = np.array(scene_paths).astype(np.string_)

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, index: int) -> Tuple[str, int]:
        scene_idx: int = (
            np.searchsorted(self.cumulative_lengths, index, side="right").item() - 1
        )

        scene_path: str = str(self.scene_paths[scene_idx], encoding="utf-8")
        scene_elem_index: int = index - self.cumulative_lengths[scene_idx].item()
        return (scene_path, scene_elem_index)
