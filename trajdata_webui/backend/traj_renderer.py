"""Convert AgentBatch → Bokeh ColumnDataSource dicts for trajectory plotting."""
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from torch.utils.data import DataLoader

from trajdata import AgentBatch, UnifiedDataset
from trajdata.data_structures.collation import agent_collate_fn
from trajdata.utils import vis_utils

# Colour used when type lookup fails
_FALLBACK_COLOR = "#888888"


def _agent_color(agent_type_val: int) -> str:
    try:
        return vis_utils.get_agent_type_color(agent_type_val)
    except Exception:
        return _FALLBACK_COLOR


def sample_to_batch(dataset: UnifiedDataset, idx: int) -> AgentBatch:
    """Load a single sample and collate it into a 1-element AgentBatch."""
    elem = dataset[idx]
    return agent_collate_fn([elem], return_dict=False, pad_format="outside")


def batch_to_sources(
    batch: AgentBatch,
    batch_idx: int = 0,
    show_hist: bool = True,
    show_fut: bool = True,
    show_neighbors: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return two ColumnDataSource data dicts: (hist_data, fut_data).

    Each dict has keys: xs, ys, line_color, line_dash, legend_label.
    """
    agent_type_val: int = batch.agent_type[batch_idx].item()
    num_neigh: int = batch.num_neigh[batch_idx].item()

    hist_xs, hist_ys, hist_colors, hist_dashes, hist_labels = [], [], [], [], []
    fut_xs, fut_ys, fut_colors, fut_dashes, fut_labels = [], [], [], [], []

    # ── ego agent ──────────────────────────────────────────────────────
    if show_hist:
        h = batch.agent_hist[batch_idx].cpu().numpy()
        hist_xs.append(_safe_xy(h, "x"))
        hist_ys.append(_safe_xy(h, "y"))
        hist_colors.append(_agent_color(agent_type_val))
        hist_dashes.append("dashed")
        hist_labels.append("Ego history")

    if show_fut:
        f = batch.agent_fut[batch_idx].cpu().numpy()
        fut_xs.append(_safe_xy(f, "x"))
        fut_ys.append(_safe_xy(f, "y"))
        fut_colors.append(_agent_color(agent_type_val))
        fut_dashes.append("solid")
        fut_labels.append("Ego future")

    # ── neighbors ──────────────────────────────────────────────────────
    if show_neighbors and num_neigh > 0:
        neigh_types = batch.neigh_types[batch_idx].cpu().numpy()
        for n in range(num_neigh):
            c = _agent_color(int(neigh_types[n]))
            if show_hist:
                nh = batch.neigh_hist[batch_idx, n].cpu().numpy()
                hist_xs.append(_safe_xy(nh, "x"))
                hist_ys.append(_safe_xy(nh, "y"))
                hist_colors.append(c)
                hist_dashes.append("dashed")
                hist_labels.append(f"Neigh {n} hist")
            if show_fut:
                nf = batch.neigh_fut[batch_idx, n].cpu().numpy()
                fut_xs.append(_safe_xy(nf, "x"))
                fut_ys.append(_safe_xy(nf, "y"))
                fut_colors.append(c)
                fut_dashes.append("solid")
                fut_labels.append(f"Neigh {n} fut")

    hist_data = dict(xs=hist_xs, ys=hist_ys,
                     line_color=hist_colors, line_dash=hist_dashes,
                     legend_label=hist_labels)
    fut_data = dict(xs=fut_xs, ys=fut_ys,
                    line_color=fut_colors, line_dash=fut_dashes,
                    legend_label=fut_labels)
    return hist_data, fut_data


def _safe_xy(state_np, attr: str) -> List[float]:
    """Return a list of floats for *attr* from a numpy StateArray; mask NaNs."""
    try:
        vals = state_np.get_attr(attr)
        return [float(v) for v in vals if not np.isnan(v)]
    except Exception:
        return []
