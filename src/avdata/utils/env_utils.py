from typing import Dict, List, Type

from avdata.dataset_specific import RawDataset
from avdata.dataset_specific.eth_ucy_peds import EUPedsDataset

try:
    from avdata.dataset_specific.lyft import LyftDataset
except ModuleNotFoundError:
    # This can happen if the user did not install avdata
    # with the "avdata[lyft]" option.
    pass

try:
    from avdata.dataset_specific.nusc import NuscDataset
except ModuleNotFoundError:
    # This can happen if the user did not install avdata
    # with the "avdata[nusc]" option.
    pass


def get_raw_dataset(dataset_name: str, data_dir: str) -> RawDataset:
    if "nusc" in dataset_name:
        return NuscDataset(dataset_name, data_dir, parallelizable=False)

    if "lyft" in dataset_name:
        return LyftDataset(dataset_name, data_dir, parallelizable=True)

    if "eupeds" in dataset_name:
        return EUPedsDataset(dataset_name, data_dir, parallelizable=True)


def get_raw_datasets(data_dirs: Dict[str, str]) -> List[RawDataset]:
    raw_datasets: List[RawDataset] = list()

    for dataset_name, data_dir in data_dirs.items():
        raw_datasets.append(get_raw_dataset(dataset_name, data_dir))

    return raw_datasets
