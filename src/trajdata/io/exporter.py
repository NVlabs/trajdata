"""
DataExporter: Precompute and save UnifiedDataset batches to disk for fast reloading.

Supported formats:
  - "zarr": Zarr compressed arrays (default, already a trajdata dependency)
  - "numpy": Uncompressed .npy files (portable, no extra deps)

Usage:
    from trajdata.io import DataExporter
    DataExporter.export(dataset, "my_cache.zarr", format="zarr", batch_size=64)
"""
import json
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import zarr
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Fields from AgentBatch that we store (tensor → numpy)
_FLOAT_FIELDS = [
    "curr_agent_state",
    "agent_hist",
    "agent_fut",
    "neigh_hist",
    "neigh_fut",
]
_INT_FIELDS = [
    "agent_hist_len",
    "agent_fut_len",
    "neigh_hist_len",
    "neigh_fut_len",
    "neigh_types",
    "agent_type",
    "scene_ts",
]
_FLOAT_SCALAR = ["dt"]
_ALL_FIELDS = _FLOAT_FIELDS + _INT_FIELDS + _FLOAT_SCALAR


class DataExporter:
    """Export a UnifiedDataset to a precomputed on-disk cache for fast reloading."""

    @staticmethod
    def export(
        dataset,
        output_path: str,
        format: str = "zarr",
        batch_size: int = 64,
        num_workers: int = 0,
        verbose: bool = True,
        compressor: Optional[object] = None,
    ) -> None:
        """Export all samples in *dataset* to *output_path*.

        Args:
            dataset: A :class:`~trajdata.UnifiedDataset` instance (agent-centric).
            output_path: Directory/file path for the exported data.
            format: ``"zarr"`` or ``"numpy"``.
            batch_size: Batch size to use during export iteration.
            num_workers: Worker count for the temporary DataLoader.
            verbose: Show progress bar.
            compressor: Custom Zarr compressor (``None`` → Zarr default Blosc).
        """
        if format not in ("zarr", "numpy"):
            raise ValueError(f"Unsupported format '{format}'. Choose 'zarr' or 'numpy'.")

        output_path = Path(output_path)

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=dataset.get_collate_fn(),
            num_workers=num_workers,
        )

        if format == "zarr":
            DataExporter._export_zarr(loader, output_path, len(dataset), verbose, compressor)
        else:
            DataExporter._export_numpy(loader, output_path, len(dataset), verbose)

        if verbose:
            logger.info("Export complete → %s", output_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _batch_to_numpy(batch) -> dict:
        """Convert an AgentBatch to a dict of numpy arrays."""
        result = {}
        for field in _ALL_FIELDS:
            tensor = getattr(batch, field, None)
            if tensor is None:
                continue
            try:
                result[field] = tensor.numpy()
            except Exception:
                pass
        # agent_name is a list of strings
        if hasattr(batch, "agent_name") and batch.agent_name is not None:
            result["agent_name"] = np.array(batch.agent_name, dtype=object)
        return result

    @staticmethod
    def _export_zarr(loader, output_path: Path, total: int, verbose: bool, compressor) -> None:
        store = zarr.open(str(output_path), mode="w")
        arrays = {}  # field → zarr Array (created on first batch)
        idx = 0

        for batch in tqdm(loader, desc="Exporting (zarr)", disable=not verbose):
            np_data = DataExporter._batch_to_numpy(batch)
            n = next(iter(np_data.values())).shape[0]

            for field, arr in np_data.items():
                if field not in arrays:
                    shape = (total,) + arr.shape[1:]
                    dtype = arr.dtype if arr.dtype != object else str
                    arrays[field] = store.zeros(
                        field,
                        shape=shape,
                        dtype=dtype,
                        chunks=(min(64, total),) + arr.shape[1:],
                        compressor=compressor,
                        object_codec=zarr.codecs.VLenUTF8() if dtype == str else None,
                    )
                try:
                    arrays[field][idx : idx + n] = arr
                except Exception:
                    pass

            idx += n

        store.attrs["total_samples"] = idx
        store.attrs["fields"] = list(arrays.keys())

    @staticmethod
    def _pad_ragged(chunks: list) -> list:
        """Pad ragged arrays along axis=1 (neighbour dimension) to the same size."""
        max_n = max(c.shape[1] for c in chunks)
        padded = []
        for c in chunks:
            pad_width = [(0, 0)] * c.ndim
            pad_width[1] = (0, max_n - c.shape[1])
            padded.append(np.pad(c, pad_width))
        return padded

    @staticmethod
    def _export_numpy(loader, output_path: Path, total: int, verbose: bool) -> None:
        output_path.mkdir(parents=True, exist_ok=True)
        buffers = {}  # field → list of batches

        for batch in tqdm(loader, desc="Exporting (numpy)", disable=not verbose):
            np_data = DataExporter._batch_to_numpy(batch)
            for field, arr in np_data.items():
                buffers.setdefault(field, []).append(arr)

        # Neighbour fields are ragged (max_neigh varies per batch) → pad to max
        _NEIGH_FIELDS = {"neigh_hist", "neigh_fut", "neigh_hist_len", "neigh_fut_len", "neigh_types"}
        for field in list(buffers.keys()):
            if field in _NEIGH_FIELDS:
                buffers[field] = DataExporter._pad_ragged(buffers[field])

        meta = {"total_samples": 0, "fields": []}
        for field, chunks in buffers.items():
            try:
                combined = np.concatenate(chunks, axis=0)
                np.save(output_path / f"{field}.npy", combined)
                meta["fields"].append(field)
                meta["total_samples"] = combined.shape[0]
            except Exception as e:
                logger.warning("Skipping field '%s': %s", field, e)

        with open(output_path / "metadata.json", "w") as f:
            json.dump(meta, f)
