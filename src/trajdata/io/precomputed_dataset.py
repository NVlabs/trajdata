"""
PrecomputedDataset: Fast PyTorch Dataset that reads from a precomputed cache.

Usage:
    from trajdata.io import DataExporter, PrecomputedDataset

    # 1. Export once
    DataExporter.export(original_dataset, "cache.zarr", format="zarr")

    # 2. Load fast (no on-the-fly preprocessing)
    fast_ds = PrecomputedDataset("cache.zarr", format="zarr")
    loader = DataLoader(fast_ds, batch_size=64, shuffle=True)
"""
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset


class PrecomputedDataset(Dataset):
    """PyTorch Dataset backed by a precomputed cache (zarr or numpy).

    Each ``__getitem__`` returns a dict of tensors corresponding to the
    fields stored by :class:`~trajdata.io.DataExporter`.
    """

    def __init__(self, path: str, format: str = "zarr", fields: Optional[List[str]] = None):
        """
        Args:
            path: Path to the exported cache (zarr store or numpy directory).
            format: ``"zarr"`` or ``"numpy"``.
            fields: Optional subset of field names to load. ``None`` loads all.
        """
        self.path = Path(path)
        self.format = format
        self._data: Dict[str, np.ndarray] = {}

        if format == "zarr":
            self._load_zarr(fields)
        elif format == "numpy":
            self._load_numpy(fields)
        else:
            raise ValueError(f"Unsupported format '{format}'")

        self._len = next(iter(self._data.values())).shape[0] if self._data else 0

    # ------------------------------------------------------------------

    def _load_zarr(self, fields: Optional[List[str]]) -> None:
        store = zarr.open(str(self.path), mode="r")
        available = list(store.attrs.get("fields", store.array_keys()))
        to_load = fields if fields is not None else available
        for field in to_load:
            if field in store:
                self._data[field] = store[field]  # lazy-loaded zarr array

    def _load_numpy(self, fields: Optional[List[str]]) -> None:
        meta_path = self.path / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            available = meta.get("fields", [])
        else:
            available = [p.stem for p in self.path.glob("*.npy")]

        to_load = fields if fields is not None else available
        for field in to_load:
            npy_path = self.path / f"{field}.npy"
            if npy_path.exists():
                self._data[field] = np.load(npy_path, allow_pickle=True)

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = {}
        for field, arr in self._data.items():
            val = arr[idx]
            if isinstance(val, np.ndarray) and val.dtype.kind in ("f", "i", "u"):
                sample[field] = torch.from_numpy(np.array(val))
            else:
                sample[field] = val  # strings, objects etc. pass through
        return sample

    @property
    def fields(self) -> List[str]:
        """List of available field names."""
        return list(self._data.keys())

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Default collate: stacks tensor fields, lists others."""
        result = {}
        for key in batch[0]:
            vals = [item[key] for item in batch]
            if isinstance(vals[0], torch.Tensor):
                try:
                    result[key] = torch.stack(vals, dim=0)
                except RuntimeError:
                    result[key] = vals
            else:
                result[key] = vals
        return result
