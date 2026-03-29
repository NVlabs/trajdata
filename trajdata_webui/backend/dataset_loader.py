"""Build a UnifiedDataset from a split name + aug config."""
from collections import defaultdict
from typing import Any, Dict, List, Optional

from trajdata import AgentType, UnifiedDataset
from trajdata.augmentation import (
    MirrorAugmentation,
    MotionTypeLabeler,
    SpeedScaleAugmentation,
)

# Datasets available without extra downloads
AVAILABLE_SPLITS = [
    "eupeds_eth-train",
    "eupeds_eth-val",
    "eupeds_hotel-train",
    "eupeds_hotel-val",
    "eupeds_univ-train",
    "eupeds_univ-val",
    "eupeds_zara1-train",
    "eupeds_zara1-val",
    "eupeds_zara2-train",
    "eupeds_zara2-val",
]

DATA_DIRS = {"eupeds_eth":   "~/datasets/eth_ucy",
             "eupeds_hotel": "~/datasets/eth_ucy",
             "eupeds_univ":  "~/datasets/eth_ucy",
             "eupeds_zara1": "~/datasets/eth_ucy",
             "eupeds_zara2": "~/datasets/eth_ucy"}


def build_augmentations(cfg: Dict[str, Any]) -> list:
    augs = []
    if cfg.get("mirror"):
        augs.append(MirrorAugmentation(axis=cfg["mirror_axis"], prob=cfg["mirror_prob"]))
    if cfg.get("speed_scale"):
        augs.append(SpeedScaleAugmentation(cfg["speed_min"], cfg["speed_max"]))
    if cfg.get("motion_labeler"):
        augs.append(MotionTypeLabeler(
            stationary_thresh=cfg["stationary_thresh"],
            walking_thresh=cfg["walking_thresh"],
            running_thresh=cfg["running_thresh"],
        ))
    return augs


def load_dataset(split: str, aug_config: Dict[str, Any]) -> UnifiedDataset:
    """Build and return a UnifiedDataset for *split*."""
    augs = build_augmentations(aug_config)
    dataset = UnifiedDataset(
        desired_data=[split],
        centric="agent",
        desired_dt=0.4,
        history_sec=(3.2, 3.2),
        future_sec=(4.8, 4.8),
        only_predict=[AgentType.PEDESTRIAN],
        augmentations=augs,
        num_workers=0,
        verbose=False,
        data_dirs=DATA_DIRS,
    )
    return dataset
