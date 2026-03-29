"""Compute summary statistics for a loaded UnifiedDataset."""
from collections import Counter
from typing import Any, Dict

import numpy as np
from torch.utils.data import DataLoader

from trajdata import AgentType, UnifiedDataset


def compute_stats(dataset: UnifiedDataset, max_batches: int = 30) -> Dict[str, Any]:
    """Return a dict of human-readable statistics for *dataset*."""
    stats: Dict[str, Any] = {}
    stats["total_samples"] = len(dataset)
    stats["num_scenes"] = dataset.num_scenes()

    # Scene info from first scene
    try:
        scene0 = dataset.get_scene(0)
        stats["dt_s"] = round(scene0.dt, 3)
        stats["scene0_name"] = scene0.name
        stats["scene0_timesteps"] = scene0.length_timesteps
        stats["scene0_agents"] = len(scene0.agents)
    except Exception:
        stats["dt_s"] = "?"
        stats["scene0_name"] = "?"
        stats["scene0_timesteps"] = "?"
        stats["scene0_agents"] = "?"

    # Agent type distribution (from a small loader sample)
    type_counter: Counter = Counter()
    hist_lens = []
    fut_lens = []

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=dataset.get_collate_fn(),
        num_workers=0,
    )
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        for t in batch.agent_type.tolist():
            try:
                type_counter[AgentType(t).name] += 1
            except ValueError:
                type_counter["UNKNOWN"] += 1
        hist_lens.extend(batch.agent_hist_len.tolist())
        fut_lens.extend(batch.agent_fut_len.tolist())

    stats["agent_type_counts"] = dict(type_counter)
    stats["mean_hist_len"] = round(float(np.mean(hist_lens)), 2) if hist_lens else 0
    stats["mean_fut_len"] = round(float(np.mean(fut_lens)), 2) if fut_lens else 0
    return stats
