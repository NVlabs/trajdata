from typing import Dict

import numpy as np
import pandas as pd


class SimMetric:
    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self, gt_df: pd.DataFrame, sim_df: pd.DataFrame) -> Dict[str, float]:
        raise NotImplementedError()


class ADE(SimMetric):
    def __init__(self) -> None:
        super().__init__("ade")

    def __call__(self, gt_df: pd.DataFrame, sim_df: pd.DataFrame) -> Dict[str, float]:
        err_df = pd.DataFrame(index=gt_df.index, columns=["error"])
        err_df["error"] = np.linalg.norm(gt_df[["x", "y"]] - sim_df[["x", "y"]], axis=1)
        return err_df.groupby("agent_id")["error"].mean().to_dict()


class FDE(SimMetric):
    def __init__(self) -> None:
        super().__init__("fde")

    def __call__(self, gt_df: pd.DataFrame, sim_df: pd.DataFrame) -> Dict[str, float]:
        err_df = pd.DataFrame(index=gt_df.index, columns=["error"])
        err_df["error"] = np.linalg.norm(gt_df[["x", "y"]] - sim_df[["x", "y"]], axis=1)
        return err_df.groupby("agent_id")["error"].last().to_dict()


class CollisionMetric(SimMetric):
    """Detect pairwise agent collisions based on Euclidean distance threshold.

    Returns per-agent collision rate (fraction of timesteps in which the
    agent is within ``distance_thresh`` of at least one other agent).

    Args:
        distance_thresh: Distance (metres) below which two agents are considered
            to have collided (default 0.5 m).
    """

    def __init__(self, distance_thresh: float = 0.5) -> None:
        super().__init__("collision_rate")
        self.distance_thresh = distance_thresh

    def __call__(self, gt_df: pd.DataFrame, sim_df: pd.DataFrame) -> Dict[str, float]:
        # sim_df index: (agent_id, scene_ts)
        results: Dict[str, float] = {}

        sim_reset = sim_df.reset_index()
        agent_ids = sim_reset["agent_id"].unique()

        for agent_id in agent_ids:
            agent_ts = sim_reset[sim_reset["agent_id"] == agent_id][
                ["scene_ts", "x", "y"]
            ].set_index("scene_ts")
            others = sim_reset[sim_reset["agent_id"] != agent_id]

            collision_ts = set()
            for _, row in others.iterrows():
                ts = row["scene_ts"]
                if ts not in agent_ts.index:
                    continue
                dx = agent_ts.loc[ts, "x"] - row["x"]
                dy = agent_ts.loc[ts, "y"] - row["y"]
                if np.sqrt(dx**2 + dy**2) < self.distance_thresh:
                    collision_ts.add(ts)

            n_ts = len(agent_ts)
            results[str(agent_id)] = len(collision_ts) / n_ts if n_ts > 0 else 0.0

        return results


class OffRoadRate(SimMetric):
    """Fraction of timesteps an agent spends outside a bounding box.

    Useful as a proxy for off-road / out-of-bounds detection when no map is
    available.  Provide ``scene_bounds`` as ``(x_min, x_max, y_min, y_max)``.

    Args:
        scene_bounds: Tuple ``(x_min, x_max, y_min, y_max)`` defining the
            valid region.  Derived automatically from the ground-truth
            trajectory if not provided.
    """

    def __init__(self, scene_bounds=None) -> None:
        super().__init__("off_road_rate")
        self.scene_bounds = scene_bounds

    def __call__(self, gt_df: pd.DataFrame, sim_df: pd.DataFrame) -> Dict[str, float]:
        if self.scene_bounds is not None:
            x_min, x_max, y_min, y_max = self.scene_bounds
        else:
            x_min = gt_df["x"].min() - 5.0
            x_max = gt_df["x"].max() + 5.0
            y_min = gt_df["y"].min() - 5.0
            y_max = gt_df["y"].max() + 5.0

        results: Dict[str, float] = {}
        sim_reset = sim_df.reset_index()

        for agent_id, grp in sim_reset.groupby("agent_id"):
            out_of_bounds = (
                (grp["x"] < x_min)
                | (grp["x"] > x_max)
                | (grp["y"] < y_min)
                | (grp["y"] > y_max)
            )
            results[str(agent_id)] = out_of_bounds.mean()

        return results
