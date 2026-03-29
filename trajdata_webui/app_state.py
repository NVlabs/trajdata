"""Shared per-session state passed between all tabs."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AppState:
    # Loaded dataset
    dataset: Optional[Any] = None
    dataset_split: str = "eupeds_eth-train"

    # Visualization
    current_sample_idx: int = 0

    # Augmentation config (mirrors widget values)
    aug_config: Dict[str, Any] = field(default_factory=lambda: {
        "mirror":          False,
        "mirror_axis":     "x",
        "mirror_prob":     0.5,
        "speed_scale":     False,
        "speed_min":       0.8,
        "speed_max":       1.2,
        "motion_labeler":  False,
        "stationary_thresh": 0.5,
        "walking_thresh":  2.5,
        "running_thresh":  6.0,
    })

    # Simulation config
    sim_config: Dict[str, Any] = field(default_factory=lambda: {
        "scene_idx": 0,
        "max_steps": 30,
        "metrics":   ["ADE", "FDE", "Collision"],
    })

    # Export config
    export_config: Dict[str, Any] = field(default_factory=lambda: {
        "output_path": "~/trajdata_export",
        "format":      "zarr",
        "batch_size":  64,
    })

    # UI State
    active_panel: str = "dashboard"
    lang: str = "en"
    theme: str = "dark"

    # Status messages (set by background threads, read by UI)
    status: str = "Ready. Load a dataset to begin."
