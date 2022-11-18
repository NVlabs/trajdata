from typing import Dict, List

from trajdata.dataset_specific.eth_ucy_peds import EUPedsDataset
from trajdata.dataset_specific.raw_dataset import RawDataset

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


try:
    from trajdata.dataset_specific.nuplan import NuplanDataset
except ModuleNotFoundError:
    # This can happen if the user did not install trajdata
    # with the "trajdata[nuplan]" option.
    pass


def get_raw_dataset(dataset_name: str, data_dir: str) -> RawDataset:
    if "nusc" in dataset_name:
        return NuscDataset(dataset_name, data_dir, parallelizable=False, has_maps=True)

    if "lyft" in dataset_name:
        return LyftDataset(dataset_name, data_dir, parallelizable=True, has_maps=True)

    if "eupeds" in dataset_name:
        return EUPedsDataset(
            dataset_name, data_dir, parallelizable=True, has_maps=False
        )

    if "nuplan" in dataset_name:
        return NuplanDataset(dataset_name, data_dir, parallelizable=True, has_maps=True)

    raise ValueError(f"Dataset with name '{dataset_name}' is not supported")


def get_raw_datasets(data_dirs: Dict[str, str]) -> List[RawDataset]:
    raw_datasets: List[RawDataset] = list()

    for dataset_name, data_dir in data_dirs.items():
        raw_datasets.append(get_raw_dataset(dataset_name, data_dir))

    return raw_datasets
