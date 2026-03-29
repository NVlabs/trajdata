"""Compute before/after augmentation trajectory sources."""
from typing import Any, Dict, Tuple

from trajdata import UnifiedDataset
from trajdata.data_structures.collation import agent_collate_fn

from .dataset_loader import build_augmentations
from .traj_renderer import batch_to_sources


def compute_preview(
    dataset: UnifiedDataset,
    sample_idx: int,
    aug_config: Dict[str, Any],
) -> Tuple[Dict, Dict, Dict, Dict]:
    """Return (base_hist, base_fut, aug_hist, aug_fut) ColumnDataSource dicts."""
    elem = dataset[sample_idx]
    base_batch = agent_collate_fn([elem], return_dict=False, pad_format="outside")
    base_hist, base_fut = batch_to_sources(base_batch)

    # Apply augmentations to a fresh copy of the batch element
    aug_batch = agent_collate_fn([elem], return_dict=False, pad_format="outside")
    for aug in build_augmentations(aug_config):
        try:
            aug.apply_agent(aug_batch)
        except Exception:
            pass
    aug_hist, aug_fut = batch_to_sources(aug_batch)

    return base_hist, base_fut, aug_hist, aug_fut
