from typing import Dict, List

from trajdata.dataset_specific import RawDataset


def get_raw_dataset(dataset_name: str, data_dir: str) -> RawDataset:
    if "nusc" in dataset_name:
        from trajdata.dataset_specific.nusc import NuscDataset

        return NuscDataset(dataset_name, data_dir, parallelizable=False, has_maps=True)

    if "lyft" in dataset_name:
        from trajdata.dataset_specific.lyft import LyftDataset

        return LyftDataset(dataset_name, data_dir, parallelizable=True, has_maps=True)

    if "eupeds" in dataset_name:
        from trajdata.dataset_specific.eth_ucy_peds import EUPedsDataset

        return EUPedsDataset(
            dataset_name, data_dir, parallelizable=True, has_maps=False
        )

    if "sdd" in dataset_name:
        from trajdata.dataset_specific.sdd_peds import SDDPedsDataset

        return SDDPedsDataset(
            dataset_name, data_dir, parallelizable=True, has_maps=False
        )

    if "nuplan" in dataset_name:
        from trajdata.dataset_specific.nuplan import NuplanDataset

        return NuplanDataset(dataset_name, data_dir, parallelizable=True, has_maps=True)

    if "waymo" in dataset_name:
        from trajdata.dataset_specific.waymo import WaymoDataset

        return WaymoDataset(dataset_name, data_dir, parallelizable=True, has_maps=True)

    if "interaction" in dataset_name:
        from trajdata.dataset_specific.interaction import InteractionDataset

        return InteractionDataset(
            dataset_name, data_dir, parallelizable=True, has_maps=True
        )

    raise ValueError(f"Dataset with name '{dataset_name}' is not supported")


def get_raw_datasets(data_dirs: Dict[str, str]) -> List[RawDataset]:
    raw_datasets: List[RawDataset] = list()

    for dataset_name, data_dir in data_dirs.items():
        raw_datasets.append(get_raw_dataset(dataset_name, data_dir))

    return raw_datasets
