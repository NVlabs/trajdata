from trajdata.data_structures import EnvMetadata
from trajdata.dataset_specific.raw_dataset import RawDataset
from trajdata.dataset_specific.waymo import waymo_utils
from typing import Any, Dict, List, Optional, Tuple, Type, Union


class WaymoDataset(RawDataset):
    def compute_metadata(self, env_name: str, data_dir: str) -> EnvMetadata:
        pass

    def load_dataset_obj(self, verbose: bool = False) -> None:
        pass