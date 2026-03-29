"""
Demo: 4 new trajdata features
=============================
1. Fast I/O  – export dataset → zarr, reload with PrecomputedDataset
2. CSV Dataset – load any CSV directory as a dataset
3. Data Enrichment – MirrorAugmentation, SpeedScaleAugmentation, MotionTypeLabeler
4. Advanced Simulation – CollisionMetric, OffRoadRate, SimRunner + ConstantVelocityPolicy
"""
import os
import tempfile
from pathlib import Path
from collections import defaultdict

import numpy as np
from torch.utils.data import DataLoader

from trajdata import AgentType, UnifiedDataset
from trajdata.augmentation import MirrorAugmentation, MotionTypeLabeler, SpeedScaleAugmentation

# ──────────────────────────────────────────────────────────────────────────────
# Shared base dataset (ETH/UCY, already downloaded)
# ──────────────────────────────────────────────────────────────────────────────
BASE_DATA_DIRS = {"eupeds_eth": "~/datasets/eth_ucy"}


def make_base_dataset(**kwargs):
    return UnifiedDataset(
        desired_data=["eupeds_eth-train"],
        centric="agent",
        desired_dt=0.4,
        history_sec=(3.2, 3.2),
        future_sec=(4.8, 4.8),
        only_predict=[AgentType.PEDESTRIAN],
        num_workers=0,
        verbose=False,
        data_dirs=BASE_DATA_DIRS,
        **kwargs,
    )


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE 1 – Fast I/O (zarr + numpy)
# ══════════════════════════════════════════════════════════════════════════════

def demo_fast_io():
    print("\n" + "="*60)
    print("FEATURE 1 – Fast I/O Formats")
    print("="*60)

    from trajdata.io import DataExporter, PrecomputedDataset

    dataset = make_base_dataset()
    print(f"  Original dataset: {len(dataset):,} samples")

    with tempfile.TemporaryDirectory() as tmpdir:
        # ── zarr export ──
        zarr_path = Path(tmpdir) / "cache.zarr"
        print("  Exporting to zarr …")
        DataExporter.export(dataset, str(zarr_path), format="zarr",
                            batch_size=32, num_workers=0, verbose=False)

        fast_ds = PrecomputedDataset(str(zarr_path), format="zarr")
        print(f"  PrecomputedDataset (zarr): {len(fast_ds):,} samples")
        sample = fast_ds[0]
        print(f"  Fields available: {fast_ds.fields}")
        if "agent_hist" in sample:
            print(f"  agent_hist shape: {sample['agent_hist'].shape}")

        # ── numpy export ──
        np_path = Path(tmpdir) / "cache_np"
        print("  Exporting to numpy …")
        DataExporter.export(dataset, str(np_path), format="numpy",
                            batch_size=32, num_workers=0, verbose=False)

        fast_np = PrecomputedDataset(str(np_path), format="numpy")
        print(f"  PrecomputedDataset (numpy): {len(fast_np):,} samples")

    print("  ✓ Feature 1 complete")


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE 2 – CSV Dataset Adapter
# ══════════════════════════════════════════════════════════════════════════════

def demo_csv_dataset():
    print("\n" + "="*60)
    print("FEATURE 2 – CSV Dataset Support")
    print("="*60)

    import json

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # ── generate synthetic CSV scene ──
        rng = np.random.default_rng(42)
        for scene_idx in range(3):
            rows = []
            for agent_id in range(5):
                x, y = rng.uniform(0, 50, 2)
                vx, vy = rng.uniform(-1, 1, 2)
                for t in range(30):
                    rows.append({
                        "frame_id": t,
                        "agent_id": agent_id,
                        "x": x + vx * t * 0.1,
                        "y": y + vy * t * 0.1,
                    })
            import pandas as pd
            pd.DataFrame(rows).to_csv(tmpdir / f"scene_{scene_idx:03d}.csv", index=False)

        # ── config.json with splits ──
        config = {
            "dt": 0.1,
            "splits": {
                "train": ["scene_000", "scene_001"],
                "val": ["scene_002"],
            }
        }
        (tmpdir / "config.json").write_text(json.dumps(config))

        # ── load via UnifiedDataset ──
        dataset = UnifiedDataset(
            desired_data=["csv_mydata-train"],
            centric="agent",
            desired_dt=0.1,
            history_sec=(1.0, 1.0),
            future_sec=(1.0, 1.0),
            only_predict=[AgentType.PEDESTRIAN],
            num_workers=0,
            verbose=False,
            data_dirs={"csv_mydata": str(tmpdir)},
        )
        print(f"  CSV dataset (train split): {len(dataset):,} samples")

        if len(dataset) > 0:
            loader2 = DataLoader(dataset, batch_size=4, shuffle=False,
                                 collate_fn=dataset.get_collate_fn(), num_workers=0)
            batch2 = next(iter(loader2))
            print(f"  Sample agent_hist shape: {batch2.agent_hist.shape}")

    print("  ✓ Feature 2 complete")


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE 3 – Data Enrichment Augmentations
# ══════════════════════════════════════════════════════════════════════════════

def demo_enrichment():
    print("\n" + "="*60)
    print("FEATURE 3 – Data Enrichment & Auto-Labeling")
    print("="*60)

    # Combine all three new augmentations
    augmentations = [
        MirrorAugmentation(axis="x", prob=0.5),
        SpeedScaleAugmentation(scale_min=0.8, scale_max=1.2),
        MotionTypeLabeler(stationary_thresh=0.3, walking_thresh=2.0),
    ]

    dataset = make_base_dataset(augmentations=augmentations)
    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=dataset.get_collate_fn(),
        num_workers=0,
    )

    batch = next(iter(loader))

    print(f"  agent_hist shape : {batch.agent_hist.shape}")
    print(f"  agent_fut  shape : {batch.agent_fut.shape}")

    if "motion_type" in batch.extras:
        mt = batch.extras["motion_type"]
        labels = {0: "STATIONARY", 1: "WALKING", 2: "RUNNING", 3: "FAST"}
        for lbl_id, lbl_name in labels.items():
            count = (mt == lbl_id).sum().item()
            print(f"    {lbl_name:12s}: {count} agents")
    else:
        print("  (motion_type not in extras – state format may lack velocity channels)")

    print("  ✓ Feature 3 complete")


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE 4 – Advanced Simulation
# ══════════════════════════════════════════════════════════════════════════════

def demo_simulation():
    print("\n" + "="*60)
    print("FEATURE 4 – Advanced Simulation Features")
    print("="*60)

    from trajdata.simulation import (
        ADE, FDE, CollisionMetric, OffRoadRate,
        SimulationScene, ConstantVelocityPolicy, SimRunner,
    )

    dataset = make_base_dataset()

    # Pick the first available scene
    loaded_scene = dataset.get_scene(0)

    print(f"  Scene: {loaded_scene.name}  ({loaded_scene.length_timesteps} timesteps)")

    sim_scene = SimulationScene(
        env_name="sim_demo",
        scene_name="sim_scene_001",
        scene=loaded_scene,
        dataset=dataset,
        init_timestep=0,
        freeze_agents=True,
    )

    policy = ConstantVelocityPolicy()
    runner = SimRunner(sim_scene, policy, max_steps=10)

    metrics = [ADE(), FDE(), CollisionMetric(distance_thresh=1.0), OffRoadRate()]
    results = runner.run(metrics=metrics, verbose=False)

    print(f"  Simulation ran for {results['steps']} steps")
    for metric_name, per_agent in results["metrics"].items():
        avg = np.mean(list(per_agent.values()))
        print(f"  {metric_name:20s}: mean={avg:.4f}")

    print("  ✓ Feature 4 complete")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    demo_fast_io()
    demo_csv_dataset()
    demo_enrichment()
    demo_simulation()

    print("\n" + "="*60)
    print("All 4 features demonstrated successfully!")
    print("="*60)
