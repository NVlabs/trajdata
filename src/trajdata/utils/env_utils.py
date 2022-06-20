from typing import Dict, List

from trajdata.dataset_specific import RawDataset
from trajdata.dataset_specific.eth_ucy_peds import EUPedsDataset

try:
    from trajdata.dataset_specific.lyft import LyftDataset
except ModuleNotFoundError:
    # This can happen if the user did not install trajdata
    # with the "trajdata[lyft]" option.
    pass

try:
    from trajdata.dataset_specific.nusc import NuscDataset
except ModuleNotFoundError:
    # This can happen if the user did not install trajdata
    # with the "trajdata[nusc]" option.
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
