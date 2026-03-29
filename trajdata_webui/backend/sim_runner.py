"""Thin wrapper around SimRunner for the web UI."""
from typing import Any, Dict, List

import numpy as np

from trajdata import UnifiedDataset
from trajdata.simulation import (
    ADE, FDE, CollisionMetric, OffRoadRate,
    SimulationScene, ConstantVelocityPolicy, SimRunner,
)


def run_simulation(
    dataset: UnifiedDataset,
    scene_idx: int,
    max_steps: int,
    metric_names: List[str],
) -> Dict[str, Any]:
    """Run a constant-velocity simulation and return formatted results."""
    scene = dataset.get_scene(scene_idx)

    sim_scene = SimulationScene(
        env_name="webui_sim",
        scene_name=f"sim_scene_{scene_idx:04d}",
        scene=scene,
        dataset=dataset,
        init_timestep=0,
        freeze_agents=True,
    )

    metric_map = {
        "ADE":       ADE(),
        "FDE":       FDE(),
        "Collision": CollisionMetric(distance_thresh=1.0),
        "OffRoad":   OffRoadRate(),
    }
    metrics = [metric_map[m] for m in metric_names if m in metric_map]

    policy = ConstantVelocityPolicy()
    runner = SimRunner(sim_scene, policy, max_steps=max_steps)
    raw = runner.run(metrics=metrics)

    # Format: list of rows {agent, metric_name: value}
    rows = []
    if "metrics" in raw:
        # Collect all agents
        all_agents = set()
        for per_agent in raw["metrics"].values():
            all_agents.update(per_agent.keys())
        for agent in sorted(all_agents):
            row: Dict[str, Any] = {"agent": agent}
            for mname, per_agent in raw["metrics"].items():
                row[mname] = round(float(per_agent.get(agent, float("nan"))), 4)
            rows.append(row)

    # Aggregate means
    means: Dict[str, float] = {}
    if "metrics" in raw:
        for mname, per_agent in raw["metrics"].items():
            vals = [v for v in per_agent.values() if not np.isnan(v)]
            means[mname] = round(float(np.mean(vals)), 4) if vals else float("nan")

    return {"rows": rows, "means": means, "steps": raw.get("steps", max_steps)}
