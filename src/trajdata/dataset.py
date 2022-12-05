import gc
import time
from collections import defaultdict
from functools import partial
from itertools import chain
from os.path import isfile
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Final,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

import dill
import numpy as np
from torch import distributed
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from trajdata import filtering
from trajdata.augmentation.augmentation import Augmentation, BatchAugmentation
from trajdata.caching import EnvCache, SceneCache, df_cache
from trajdata.data_structures import (
    AgentBatchElement,
    AgentDataIndex,
    AgentMetadata,
    AgentType,
    DataIndex,
    Scene,
    SceneBatchElement,
    SceneDataIndex,
    SceneMetadata,
    SceneTag,
    SceneTime,
    SceneTimeAgent,
    agent_collate_fn,
    scene_collate_fn,
)
from trajdata.dataset_specific import RawDataset
from trajdata.maps import VectorMap
from trajdata.maps.map_api import MapAPI
from trajdata.parallel import ParallelDatasetPreprocessor, scene_paths_collate_fn
from trajdata.utils import agent_utils, env_utils, scene_utils, string_utils
from trajdata.utils.parallel_utils import parallel_iapply

# TODO(bivanovic): Move this to a better place in the codebase.
DEFAULT_PX_PER_M: Final[float] = 2.0


class UnifiedDataset(Dataset):
    # @profile
    def __init__(
        self,
        desired_data: List[str],
        scene_description_contains: Optional[List[str]] = None,
        centric: str = "agent",
        desired_dt: Optional[float] = None,
        history_sec: Tuple[Optional[float], Optional[float]] = (
            None,
            None,
        ),  # Both inclusive
        future_sec: Tuple[Optional[float], Optional[float]] = (
            None,
            None,
        ),  # Both inclusive
        agent_interaction_distances: Dict[
            Tuple[AgentType, AgentType], float
        ] = defaultdict(lambda: np.inf),
        incl_robot_future: bool = False,
        incl_raster_map: bool = False,
        raster_map_params: Optional[Dict[str, Any]] = None,
        incl_vector_map: bool = False,
        vector_map_params: Optional[Dict[str, Any]] = None,
        require_map_cache: bool = True,
        only_types: Optional[List[AgentType]] = None,
        only_predict: Optional[List[AgentType]] = None,
        no_types: Optional[List[AgentType]] = None,
        standardize_data: bool = True,
        standardize_derivatives: bool = False,
        augmentations: Optional[List[Augmentation]] = None,
        max_agent_num: Optional[int] = None,
        max_neighbor_num: Optional[int] = None,
        ego_only: Optional[bool] = False,
        data_dirs: Dict[str, str] = {
            "eupeds_eth": "~/datasets/eth_ucy_peds",
            "eupeds_hotel": "~/datasets/eth_ucy_peds",
            "eupeds_univ": "~/datasets/eth_ucy_peds",
            "eupeds_zara1": "~/datasets/eth_ucy_peds",
            "eupeds_zara2": "~/datasets/eth_ucy_peds",
            "nusc_mini": "~/datasets/nuScenes",
            # "nusc_trainval": "~/datasets/nuScenes",
            # "nusc_test": "~/datasets/nuScenes",
            "lyft_sample": "~/datasets/lyft/scenes/sample.zarr",
            # "lyft_train": "~/datasets/lyft/scenes/train.zarr",
            # "lyft_train_full": "~/datasets/lyft/scenes/train_full.zarr",
            # "lyft_val": "~/datasets/lyft/scenes/validate.zarr",
            # "nuplan_mini": "~/datasets/nuplan/dataset/nuplan-v1.1",
        },
        cache_type: str = "dataframe",
        cache_location: str = "~/.unified_data_cache",
        rebuild_cache: bool = False,
        rebuild_maps: bool = False,
        num_workers: int = 0,
        verbose: bool = False,
        extras: Dict[str, Callable[..., np.ndarray]] = dict(),
        transforms: Iterable[
            Callable[..., Union[AgentBatchElement, SceneBatchElement]]
        ] = (),
        rank: int = 0,
    ) -> None:
        """Instantiates a PyTorch Dataset object which aggregates data
        from multiple trajectory forecasting datasets.

        Args:
            desired_data (List[str]): Names of datasets, splits, scene tags, etc. See the README for more information.
            scene_description_contains (Optional[List[str]], optional): Only return data from scenes whose descriptions contain one or more of these strings. Defaults to None.
            centric (str, optional): One of {"agent", "scene"}, specifies what a batch element contains data for (one agent at one timestep or all agents in a scene at one timestep). Defaults to "agent".
            desired_dt (Optional[float], optional): Specifies the desired data sampling rate, an error will be raised if the original and desired data sampling rate are not integer multiples of each other. Defaults to None.
            history_sec (Tuple[Optional[float], Optional[float]], optional): A tuple containing (the minimum seconds of history each batch element must contain, the maximum seconds of history to return). Both inclusive. Defaults to ( None, None, ).
            future_sec (Tuple[Optional[float], Optional[float]], optional): A tuple containing (the minimum seconds of future data each batch element must contain, the maximum seconds of future data to return). Both inclusive. Defaults to ( None, None, ).
            agent_interaction_distances: (Dict[Tuple[AgentType, AgentType], float]): A dictionary mapping agent-agent interaction distances in meters (determines which agents are included as neighbors to the predicted agent). Defaults to infinity for all types.
            incl_robot_future (bool, optional): Include the ego agent's future trajectory in batches (accordingly, never predict the ego's future). Defaults to False.
            incl_raster_map (bool, optional): Include a local cropping of the rasterized map (if the dataset provides a map) per agent. Defaults to False.
            raster_map_params (Optional[Dict[str, Any]], optional): Local map cropping parameters, must be specified if incl_map is True. Must contain keys {"px_per_m", "map_size_px"} and can optionally contain {"offset_frac_xy"}. Defaults to None.
            incl_vector_map (bool, optional): Include information about the scene's vector map (e.g., for use in nearest lane queries as an `extras` batch element function),
            vector_map_params (Optional[Dict[str, Any]], optional): Vector map loading parameters. Defaults to None (by default only road lanes will be loaded as part of the map).
            require_map_cache (bool, optional): Cache map objects (if the dataset provides a map) regardless of the value of incl_map. Defaults to True.
            only_types (Optional[List[AgentType]], optional): Filter out all agents EXCEPT for those of the specified types. Defaults to None.
            only_predict (Optional[List[AgentType]], optional): Only predict the specified types of agents. Importantly, this keeps other agent types in the scene, e.g., as neighbors of the agent to be predicted. Defaults to None.
            no_types (Optional[List[AgentType]], optional): Filter out all agents with the specified types. Defaults to None.
            standardize_data (bool, optional): Standardize all data such that (1) the predicted agent's orientation at the current timestep is 0, (2) all data is made relative to the predicted agent's current position, and (3) the agent's heading value is replaced with its sin, cos values. Defaults to True.
            standardize_derivatives (bool, optional): Make agent velocities and accelerations relative to the agent being predicted. Defaults to False.
            augmentations (Optional[List[Augmentation]], optional): Perform the specified augmentations to the batch or dataset. Defaults to None.
            max_agent_num (int, optional): The maximum number of agents to include in a batch for scene-centric batching.
            max_neighbor_num (int, optional): The maximum number of neighbors to include in a batch for agent-centric batching.
            ego_only (bool, optional): If True, only return batches where the ego-agent is the one being predicted.
            data_dirs (Optional[Dict[str, str]], optional): Dictionary mapping dataset names to their directories on disk. Defaults to { "eupeds_eth": "~/datasets/eth_ucy_peds", "eupeds_hotel": "~/datasets/eth_ucy_peds", "eupeds_univ": "~/datasets/eth_ucy_peds", "eupeds_zara1": "~/datasets/eth_ucy_peds", "eupeds_zara2": "~/datasets/eth_ucy_peds", "nusc_mini": "~/datasets/nuScenes", "lyft_sample": "~/datasets/lyft/scenes/sample.zarr", }.
            cache_type (str, optional): What type of cache to use to store preprocessed, cached data on disk. Defaults to "dataframe".
            cache_location (str, optional): Where to store and load preprocessed, cached data. Defaults to "~/.unified_data_cache".
            rebuild_cache (bool, optional): If True, process and cache trajectory data even if it is already cached. Defaults to False.
            rebuild_maps (bool, optional): If True, process and cache maps even if they are already cached. Defaults to False.
            num_workers (int, optional): Number of parallel workers to use for dataset preprocessing and loading. Defaults to 0.
            verbose (bool, optional):  If True, print internal data loading information. Defaults to False.
            extras (Dict[str, Callable[..., np.ndarray]], optional): Adds extra data to each batch element. Each Callable must take as input a filled {Agent,Scene}BatchElement and return an ndarray which will subsequently be added to the batch element's `extra` dict.
            transforms (Iterable[Callable], optional): Allows for custom modifications of batch elements. Each Callable must take in a filled {Agent,Scene}BatchElement and return a {Agent,Scene}BatchElement.
            rank (int, optional): Proccess rank when using torch DistributedDataParallel for multi-GPU training. Only the rank 0 process will be used for caching.
        """
        self.centric: str = centric
        self.desired_dt: float = desired_dt

        if cache_type == "dataframe":
            self.cache_class = df_cache.DataFrameCache

        self.rebuild_cache: bool = rebuild_cache
        self.cache_path: Path = Path(cache_location).expanduser().resolve()
        self.cache_path.mkdir(parents=True, exist_ok=True)
        self.env_cache: EnvCache = EnvCache(self.cache_path)

        if incl_raster_map:
            assert (
                raster_map_params is not None
            ), r"Path size information, i.e., {'px_per_m': ..., 'map_size_px': ...}, must be provided if incl_map=True"
            assert (
                raster_map_params["map_size_px"] % 2 == 0
            ), "Patch parameter 'map_size_px' must be divisible by 2"

        require_map_cache = require_map_cache or incl_raster_map

        self.history_sec = history_sec
        self.future_sec = future_sec
        self.agent_interaction_distances = agent_interaction_distances
        self.incl_robot_future = incl_robot_future
        self.incl_raster_map = incl_raster_map
        self.raster_map_params = (
            raster_map_params
            if raster_map_params is not None
            else {"px_per_m": DEFAULT_PX_PER_M}
        )
        self.incl_vector_map = incl_vector_map
        self.vector_map_params = (
            vector_map_params
            if vector_map_params is not None
            else {
                "incl_road_lanes": True,
                "incl_road_areas": False,
                "incl_ped_crosswalks": False,
                "incl_ped_walkways": False,
                # Collation can be quite slow if vector maps are included,
                # so we do not unless the user requests it.
                "no_collate": True,
            }
        )
        self.only_types = None if only_types is None else set(only_types)
        self.only_predict = None if only_predict is None else set(only_predict)
        self.no_types = None if no_types is None else set(no_types)
        self.standardize_data = standardize_data
        self.standardize_derivatives = standardize_derivatives
        self.augmentations = augmentations
        self.extras = extras
        self.transforms = transforms
        self.verbose = verbose
        self.max_agent_num = max_agent_num
        self.max_neighbor_num = max_neighbor_num
        self.ego_only = ego_only

        # Ensuring scene description queries are all lowercase
        if scene_description_contains is not None:
            scene_description_contains = [s.lower() for s in scene_description_contains]

        self.envs: List[RawDataset] = env_utils.get_raw_datasets(data_dirs)
        self.envs_dict: Dict[str, RawDataset] = {env.name: env for env in self.envs}

        matching_datasets: List[SceneTag] = self.get_matching_scene_tags(desired_data)
        if self.verbose:
            print(
                "Loading data for matched scene tags:",
                string_utils.pretty_string_tags(matching_datasets),
                flush=True,
            )

        self._map_api: Optional[MapAPI] = None
        if self.incl_vector_map:
            self._map_api = MapAPI(self.cache_path)

        all_scenes_list: Union[List[SceneMetadata], List[Scene]] = list()
        for env in self.envs:
            if any(env.name in dataset_tuple for dataset_tuple in matching_datasets):
                all_data_cached: bool = False
                all_maps_cached: bool = not env.has_maps or not require_map_cache
                if self.env_cache.env_is_cached(env.name) and not self.rebuild_cache:
                    scenes_list: List[Scene] = self.get_desired_scenes_from_env(
                        matching_datasets, scene_description_contains, env
                    )

                    all_data_cached: bool = all(
                        self.env_cache.scene_is_cached(
                            scene.env_name, scene.name, scene.dt
                        )
                        for scene in scenes_list
                    )

                    all_maps_cached: bool = (
                        not env.has_maps
                        or not require_map_cache
                        or all(
                            self.cache_class.is_map_cached(
                                self.cache_path,
                                env.name,
                                scene.location,
                                self.raster_map_params["px_per_m"],
                            )
                            for scene in scenes_list
                        )
                    )

                if (
                    not all_data_cached
                    or not all_maps_cached
                    or self.rebuild_cache
                    or rebuild_maps
                ):
                    # Loading dataset objects in case we don't have
                    # the desired data already cached.
                    env.load_dataset_obj(verbose=self.verbose)

                    if (
                        rebuild_maps
                        or not all_maps_cached
                        or not self.cache_class.are_maps_cached(
                            self.cache_path, env.name
                        )
                    ):
                        # Use only rank 0 process for caching when using multi-GPU torch training.
                        if rank == 0:
                            env.cache_maps(
                                self.cache_path,
                                self.cache_class,
                                self.raster_map_params,
                            )

                        # Wait for rank 0 process to be done with caching.
                        if (
                            distributed.is_initialized()
                            and distributed.get_world_size() > 1
                        ):
                            distributed.barrier()

                    scenes_list: List[SceneMetadata] = self.get_desired_scenes_from_env(
                        matching_datasets, scene_description_contains, env
                    )

                if self.incl_vector_map:
                    for map_name in env.metadata.map_locations:
                        self._map_api.get_map(f"{env.name}:{map_name}")

                all_scenes_list += scenes_list

        # List of cached scene paths.
        scene_paths: List[Path] = self.preprocess_scene_data(
            all_scenes_list, num_workers
        )
        if self.verbose:
            print(len(scene_paths), "scenes in the scene index.")

        # Done with this list. Cutting memory usage because
        # of multiprocessing later on.
        del all_scenes_list

        data_index: Union[
            List[Tuple[str, int, np.ndarray]],
            List[Tuple[str, int, List[Tuple[str, np.ndarray]]]],
        ] = self.get_data_index(num_workers, scene_paths)

        # Done with this list. Cutting memory usage because
        # of multiprocessing later on.
        del scene_paths

        self._scene_index: List[Path] = [orig_path for orig_path, _, _ in data_index]

        # The data index is effectively a big list of tuples taking the form:
        # (scene_path: str, index_len: int, valid_timesteps: np.ndarray[, agent_name: str])
        self._data_index: DataIndex = (
            AgentDataIndex(data_index, self.verbose)
            if self.centric == "agent"
            else SceneDataIndex(data_index, self.verbose)
        )
        self._data_len: int = len(self._data_index)

        self._cached_batch_elements = None

    def load_or_create_cache(
        self, cache_path: str, num_workers=0, filter_fn=None
    ) -> None:
        if isfile(cache_path):
            print(f"Loading cache from {cache_path} ...", end="")
            t = time.time()
            with open(cache_path, "rb") as f:
                self._cached_batch_elements, keep_mask = dill.load(f, encoding="latin1")
            print(f" done in {time.time() - t:.1f}s.")

        else:
            # Build cache
            cached_batch_elements = []
            keep_mask = []

            if num_workers <= 0:
                cache_data_iterator = self
            else:
                # Use DataLoader as a generic multiprocessing framework.
                # We set batchsize=1 and a custom collate function.
                # In effect this will just call self.__getitem__ in parallel.
                cache_data_iterator = DataLoader(
                    self,
                    batch_size=1,
                    num_workers=num_workers,
                    shuffle=False,
                    collate_fn=lambda xlist: xlist[0],
                )

            for element in tqdm(
                cache_data_iterator,
                desc=f"Caching batch elements ({num_workers} CPUs): ",
                disable=False,
            ):
                if filter_fn is None or filter_fn(element):
                    cached_batch_elements.append(element)
                    keep_mask.append(True)
                else:
                    keep_mask.append(False)

            # Just deletes the variable cache_data_iterator,
            # not self (in case it is set to that)!
            del cache_data_iterator

            print(f"Saving cache to {cache_path} ....", end="")
            t = time.time()
            with open(cache_path, "wb") as f:
                dill.dump((cached_batch_elements, keep_mask), f)
            print(f" done in {time.time() - t:.1f}s.")

            self._cached_batch_elements = cached_batch_elements

        # Verify
        if len(keep_mask) != self._data_len:
            raise ValueError("Current data and keep_mask lengths do not match!")

        # Remove unwanted elements
        self.remove_elements(keep_mask=keep_mask)

        # Verify
        if len(self._cached_batch_elements) != self._data_len:
            raise ValueError("Current data and cached data lengths do not match!")

    def apply_filter(
        self,
        filter_fn: Callable[[Union[AgentBatchElement, SceneBatchElement]], bool],
        num_workers: int = 0,
    ) -> None:
        keep_mask = []

        if num_workers <= 0:
            cache_data_iterator = self
        else:
            # Use DataLoader as a generic multiprocessing framework.
            # We set batchsize=1 and a custom collate function.
            # In effect this will just call self.__getitem__ in parallel.
            cache_data_iterator = DataLoader(
                self,
                batch_size=1,
                num_workers=num_workers,
                shuffle=False,
                collate_fn=lambda xlist: xlist[0],
            )

        for element in tqdm(
            cache_data_iterator,
            desc=f"Filtering dataset ({num_workers} CPUs): ",
            disable=False,
        ):
            if filter_fn is None or filter_fn(element):
                keep_mask.append(True)
            else:
                keep_mask.append(False)

        # Just deletes the variable cache_data_iterator,
        # not self (in case it is set to that)!
        del cache_data_iterator

        # Verify
        if len(keep_mask) != self._data_len:
            raise ValueError("Current data and keep_mask lengths do not match!")

        # Remove unwanted elements
        self.remove_elements(keep_mask=keep_mask)

    def remove_elements(self, keep_mask: List[bool]):
        assert len(keep_mask) == self._data_len
        old_len = self._data_len
        self._data_index = [
            self._data_index[i] for i in range(len(keep_mask)) if keep_mask[i]
        ]
        self._data_len = len(self._data_index)

        print(
            f"Kept {self._data_len}/{old_len} elements, {self._data_len/old_len*100.0:.2f}%."
        )

    def get_data_index(
        self, num_workers: int, scene_paths: List[Path]
    ) -> Union[
        List[Tuple[str, int, np.ndarray]],
        List[Tuple[str, int, List[Tuple[str, np.ndarray]]]],
    ]:
        # We're doing all this staticmethod malarkey so that multiprocessing
        # doesn't copy the UnifiedDataset self object (which generally slows down the
        # rate of spinning up new processes and hogs memory).
        desc: str = f"Creating {self.centric.capitalize()} Data Index"

        if self.centric == "scene":
            data_index_fn = partial(
                UnifiedDataset._get_data_index_scene,
                only_types=self.only_types,
                no_types=self.no_types,
                history_sec=self.history_sec,
                future_sec=self.future_sec,
                desired_dt=self.desired_dt,
            )
        elif self.centric == "agent":
            data_index_fn = partial(
                UnifiedDataset._get_data_index_agent,
                incl_robot_future=self.incl_robot_future,
                only_types=self.only_types,
                only_predict=self.only_predict,
                no_types=self.no_types,
                history_sec=self.history_sec,
                future_sec=self.future_sec,
                desired_dt=self.desired_dt,
                ego_only=self.ego_only,
            )
        else:
            raise ValueError(f"{self.centric}-centric data batches are not supported.")

        # data_index is either:
        # [(scene_path, total_index_len, valid_scene_ts)] for scene-centric data, or
        # [(scene_path, total_index_len, [(agent_name, valid_agent_ts)])] for agent-centric data
        data_index: Union[
            List[Tuple[str, int, np.ndarray]],
            List[Tuple[str, int, List[Tuple[str, np.ndarray]]]],
        ] = list()
        if num_workers <= 1:
            for scene_info_path in tqdm(
                scene_paths,
                desc=desc + " (Serially)",
                disable=not self.verbose,
            ):
                _, orig_path, index_elems_len, index_elems = data_index_fn(
                    scene_info_path
                )
                if len(index_elems) > 0:
                    data_index.append((str(orig_path), index_elems_len, index_elems))
        else:
            for (_, orig_path, index_elems_len, index_elems) in parallel_iapply(
                data_index_fn,
                scene_paths,
                num_workers=num_workers,
                desc=desc + f" ({num_workers} CPUs)",
                disable=not self.verbose,
            ):
                if len(index_elems) > 0:
                    data_index.append((str(orig_path), index_elems_len, index_elems))

        return data_index

    @staticmethod
    def _get_data_index_scene(
        scene_info_path: Path,
        only_types: Optional[Set[AgentType]],
        no_types: Optional[Set[AgentType]],
        history_sec: Tuple[Optional[float], Optional[float]],
        future_sec: Tuple[Optional[float], Optional[float]],
        desired_dt: Optional[float],
        ret_scene_info: bool = False,
    ) -> Tuple[Optional[Scene], Path, int, np.ndarray]:
        index_elems: List[int] = list()

        scene: Scene = EnvCache.load(scene_info_path)
        scene_utils.enforce_desired_dt(scene, desired_dt)

        for ts in range(scene.length_timesteps):
            # This is where we remove scene timesteps that would have no remaining agents after filtering.
            if filtering.all_agents_excluded_types(no_types, scene.agent_presence[ts]):
                continue
            elif filtering.no_agent_included_types(
                only_types, scene.agent_presence[ts]
            ):
                continue

            if filtering.no_agent_satisfies_time(
                ts,
                scene.dt,
                history_sec,
                future_sec,
                scene.agent_presence[ts],
            ):
                # Ignore this datum if no agent in the scene satisfies our time requirements.
                continue

            index_elems.append(ts)

        return (
            (scene if ret_scene_info else None),
            scene_info_path,
            len(index_elems),
            np.array(index_elems, dtype=int),
        )

    @staticmethod
    def _get_data_index_agent(
        scene_info_path: Path,
        incl_robot_future: bool,
        only_types: Optional[Set[AgentType]],
        only_predict: Optional[Set[AgentType]],
        no_types: Optional[Set[AgentType]],
        history_sec: Tuple[Optional[float], Optional[float]],
        future_sec: Tuple[Optional[float], Optional[float]],
        desired_dt: Optional[float],
        ret_scene_info: bool = False,
        ego_only: bool = False,
    ) -> Tuple[Optional[Scene], Path, int, List[Tuple[str, np.ndarray]]]:
        index_elems_len: int = 0
        index_elems: List[Tuple[str, np.ndarray]] = list()

        scene: Scene = EnvCache.load(scene_info_path)
        scene_utils.enforce_desired_dt(scene, desired_dt)

        filtered_agents: List[AgentMetadata] = filtering.agent_types(
            scene.agents,
            no_types,
            only_predict if only_predict is not None else only_types,
        )

        for agent_info in filtered_agents:
            # Don't want to predict the ego if we're going to be
            # giving the model its future!
            if incl_robot_future and agent_info.name == "ego":
                continue

            if ego_only and agent_info.name != "ego":
                # We only want to return the ego.
                continue

            valid_ts: Tuple[int, int] = filtering.get_valid_ts(
                agent_info, scene.dt, history_sec, future_sec
            )

            num_agent_ts: int = valid_ts[1] - valid_ts[0] + 1
            if num_agent_ts > 0:
                index_elems_len += num_agent_ts
                index_elems.append((agent_info.name, np.array(valid_ts, dtype=int)))

        return (
            (scene if ret_scene_info else None),
            scene_info_path,
            index_elems_len,
            index_elems,
        )

    def get_collate_fn(
        self, return_dict: bool = False, pad_format: str = "outside"
    ) -> Callable:
        if pad_format not in {"outside", "right"}:
            raise ValueError("Padding format must be one of {'outside', 'right'}")

        batch_augments: Optional[List[BatchAugmentation]] = None
        if self.augmentations:
            batch_augments = [
                batch_aug
                for batch_aug in self.augmentations
                if isinstance(batch_aug, BatchAugmentation)
            ]

        if self.centric == "agent":
            collate_fn = partial(
                agent_collate_fn,
                return_dict=return_dict,
                pad_format=pad_format,
                batch_augments=batch_augments,
            )
        elif self.centric == "scene":
            collate_fn = partial(
                scene_collate_fn,
                return_dict=return_dict,
                pad_format=pad_format,
                batch_augments=batch_augments,
            )
        else:
            raise ValueError(f"{self.centric}-centric data batches are not supported.")

        return collate_fn

    def get_matching_scene_tags(self, queries: List[str]) -> List[SceneTag]:
        # if queries is None:
        #     return list(chain.from_iterable(env.components for env in self.envs))

        query_tuples = [set(data.split("-")) for data in queries]

        matching_scene_tags: List[SceneTag] = list()
        for idx, query_tuple in enumerate(query_tuples):
            matched_tags: List[SceneTag] = list()
            for env in self.envs:
                matched_tags += env.get_matching_scene_tags(query_tuple)

            if len(matched_tags) == 0:
                raise ValueError(
                    f"No matched tags for {queries[idx]}, please ensure the relevant dataset is in data_dirs."
                )

            matching_scene_tags += matched_tags

        return matching_scene_tags

    def get_desired_scenes_from_env(
        self,
        scene_tags: List[SceneTag],
        scene_description_contains: Optional[List[str]],
        env: RawDataset,
    ) -> Union[List[Scene], List[SceneMetadata]]:
        scenes_list: Union[List[Scene], List[SceneMetadata]] = list()
        for scene_tag in scene_tags:
            if env.name in scene_tag:
                scenes_list += env.get_matching_scenes(
                    scene_tag,
                    scene_description_contains,
                    self.env_cache,
                    self.rebuild_cache,
                )

        return scenes_list

    def preprocess_scene_data(
        self,
        scenes_list: Union[List[SceneMetadata], List[Scene]],
        num_workers: int,
    ) -> List[Path]:
        all_cached: bool = not self.rebuild_cache and all(
            self.env_cache.scene_is_cached(
                scene_info.env_name,
                scene_info.name,
                self.desired_dt if self.desired_dt is not None else scene_info.dt,
            )
            for scene_info in scenes_list
        )

        serial_scenes: List[SceneMetadata]
        parallel_scenes: List[SceneMetadata]
        if num_workers > 1 and not all_cached:
            serial_scenes = [
                scene_info
                for scene_info in scenes_list
                if not self.envs_dict[scene_info.env_name].parallelizable
            ]
            parallel_scenes = [
                scene_info
                for scene_info in scenes_list
                if self.envs_dict[scene_info.env_name].parallelizable
            ]
        else:
            serial_scenes = scenes_list
            parallel_scenes = list()

        # List of (Original cached path, Temporary cached path)
        scene_paths: List[Path] = list()
        if serial_scenes:
            # Scenes for which it's faster to process them serially. See
            # the longer comment below for a more thorough explanation.
            scene_info: SceneMetadata
            for scene_info in tqdm(
                serial_scenes,
                desc="Calculating Agent Data (Serially)",
                disable=not self.verbose,
            ):
                scene_dt: float = (
                    self.desired_dt if self.desired_dt is not None else scene_info.dt
                )
                if not self.rebuild_cache and self.env_cache.scene_is_cached(
                    scene_info.env_name, scene_info.name, scene_dt
                ):
                    # This is a fast path in case we don't need to
                    # perform any modifications to the scene_info.
                    scene_path: Path = EnvCache.scene_metadata_path(
                        self.cache_path,
                        scene_info.env_name,
                        scene_info.name,
                        scene_dt,
                    )

                    scene_paths.append(scene_path)
                    continue

                corresponding_env: RawDataset = self.envs_dict[scene_info.env_name]
                scene: Scene = agent_utils.get_agent_data(
                    scene_info,
                    corresponding_env,
                    self.env_cache,
                    self.rebuild_cache,
                    self.cache_class,
                    self.desired_dt,
                )

                scene_path: Path = EnvCache.scene_metadata_path(
                    self.cache_path, scene.env_name, scene.name, scene.dt
                )
                scene_paths.append(scene_path)

        # Done with these lists. Cutting memory usage because
        # of multiprocessing below.
        del serial_scenes
        scenes_list.clear()

        # No more need for the original dataset objects and freeing up
        # this memory allows the parallel processing below to run very fast.
        # The dataset objects for any envs used below will be loaded in each
        # process.
        for env in self.envs:
            env.del_dataset_obj()

        # Scenes for which it's faster to process them in parallel
        # Note this really only applies to scenes whose raw datasets
        # are "parallelizable" AKA take up a small amount of memory
        # and effectively act as a window into the data on disk.
        # E.g., NuScenes objects load a lot of data into RAM, so
        # they are not parallelizable and should be processed
        # serially after loading the dataset object once
        # (thankfully it is quite fast to do so).
        if parallel_scenes:
            # Here we're using PyTorch's parallel dataloading as a
            # general parallel processing interface (it uses all the same
            # multiprocessing package under the hood anyways, but it has
            # some good logic for keeping workers occupied which seems
            # like it'd be good to reuse).
            parallel_preprocessor = ParallelDatasetPreprocessor(
                parallel_scenes,
                {
                    env_name: str(env.metadata.data_dir)
                    for env_name, env in self.envs_dict.items()
                },
                str(self.env_cache.path),
                self.desired_dt,
                self.cache_class,
                self.rebuild_cache,
            )

            # Done with this list. Cutting memory usage because
            # of multiprocessing below.
            del parallel_scenes

            # This shouldn't be necessary, but sometimes old
            # (large) dataset objects haven't been garbage collected
            # by this time, causing memory usage to skyrocket during
            # parallel data preprocessing below.
            gc.collect()

            dataloader = DataLoader(
                parallel_preprocessor,
                batch_size=1,
                num_workers=num_workers,
                shuffle=False,
                collate_fn=scene_paths_collate_fn,
            )

            for processed_scene_paths in tqdm(
                dataloader,
                desc=f"Calculating Agent Data ({num_workers} CPUs)",
                disable=not self.verbose,
            ):
                scene_paths += [
                    Path(path_str)
                    for path_str in processed_scene_paths
                    if path_str is not None
                ]

        return scene_paths

    def get_scene(self, scene_idx: int) -> Scene:
        scene: Scene = EnvCache.load(self._scene_index[scene_idx])
        scene_utils.enforce_desired_dt(scene, self.desired_dt)
        return scene

    def num_scenes(self) -> int:
        return len(self._scene_index)

    def scenes(self) -> Scene:
        for scene_idx in range(self.num_scenes()):
            yield self.get_scene(scene_idx)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __len__(self) -> int:
        return self._data_len

    def __getitem__(self, idx: int) -> Union[SceneBatchElement, AgentBatchElement]:
        if self._cached_batch_elements is not None:
            return self._cached_batch_elements[idx]

        if self.centric == "scene":
            scene_path, ts = self._data_index[idx]
        elif self.centric == "agent":
            scene_path, agent_id, ts = self._data_index[idx]

        scene: Scene = EnvCache.load(scene_path)
        scene_utils.enforce_desired_dt(scene, self.desired_dt)
        scene_cache: SceneCache = self.cache_class(
            self.cache_path, scene, self.augmentations
        )

        if self.centric == "scene":
            scene_time: SceneTime = SceneTime.from_cache(
                scene,
                ts,
                scene_cache,
                only_types=self.only_types,
                no_types=self.no_types,
            )

            batch_element: SceneBatchElement = SceneBatchElement(
                scene_cache,
                idx,
                scene_time,
                self.history_sec,
                self.future_sec,
                self.agent_interaction_distances,
                self.incl_robot_future,
                self.incl_raster_map,
                self.raster_map_params,
                self._map_api,
                self.vector_map_params,
                self.standardize_data,
                self.standardize_derivatives,
                self.max_agent_num,
            )
        elif self.centric == "agent":
            scene_time_agent: SceneTimeAgent = SceneTimeAgent.from_cache(
                scene,
                ts,
                agent_id,
                scene_cache,
                only_types=self.only_types,
                no_types=self.no_types,
                incl_robot_future=self.incl_robot_future,
            )

            batch_element: AgentBatchElement = AgentBatchElement(
                scene_cache,
                idx,
                scene_time_agent,
                self.history_sec,
                self.future_sec,
                self.agent_interaction_distances,
                self.incl_robot_future,
                self.incl_raster_map,
                self.raster_map_params,
                self._map_api,
                self.vector_map_params,
                self.standardize_data,
                self.standardize_derivatives,
                self.max_neighbor_num,
            )

        for key, extra_fn in self.extras.items():
            batch_element.extras[key] = extra_fn(batch_element)

        for transform_fn in self.transforms:
            batch_element = transform_fn(batch_element)

        if self.vector_map_params.get("no_collate", True):
            batch_element.vec_map = None

        return batch_element
