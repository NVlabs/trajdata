import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
from nuscenes.eval.prediction.splits import NUM_IN_TRAIN_VAL
from nuscenes.map_expansion import arcline_path_utils
from nuscenes.map_expansion.map_api import NuScenesMap, locations
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from scipy.spatial.distance import cdist
from tqdm import tqdm

from trajdata.caching import EnvCache, SceneCache
from trajdata.data_structures.agent import (
    Agent,
    AgentMetadata,
    AgentType,
    FixedExtent,
    VariableExtent,
)
from trajdata.data_structures.environment import EnvMetadata
from trajdata.data_structures.scene_metadata import Scene, SceneMetadata
from trajdata.data_structures.scene_tag import SceneTag
from trajdata.dataset_specific.nusc import nusc_utils
from trajdata.dataset_specific.raw_dataset import RawDataset
from trajdata.dataset_specific.scene_records import NuscSceneRecord
from trajdata.maps import RasterizedMap, RasterizedMapMetadata, map_utils
from trajdata.proto.vectorized_map_pb2 import (
    MapElement,
    PedCrosswalk,
    PedWalkway,
    Polyline,
    RoadArea,
    RoadLane,
    VectorizedMap,
)


class NuscDataset(RawDataset):
    def compute_metadata(self, env_name: str, data_dir: str) -> EnvMetadata:
        # We're using the nuScenes prediction challenge split here.
        # See https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/prediction/splits.py
        # for full details on how the splits are obtained below.
        all_scene_splits: Dict[str, List[str]] = create_splits_scenes()

        train_scenes: List[str] = deepcopy(all_scene_splits["train"])
        all_scene_splits["train"] = train_scenes[NUM_IN_TRAIN_VAL:]
        all_scene_splits["train_val"] = train_scenes[:NUM_IN_TRAIN_VAL]

        if env_name == "nusc_trainval":
            nusc_scene_splits: Dict[str, List[str]] = {
                k: all_scene_splits[k] for k in ["train", "train_val", "val"]
            }

            # nuScenes possibilities are the Cartesian product of these
            dataset_parts: List[Tuple[str, ...]] = [
                ("train", "train_val", "val"),
                ("boston", "singapore"),
            ]
        elif env_name == "nusc_test":
            nusc_scene_splits: Dict[str, List[str]] = {
                k: all_scene_splits[k] for k in ["test"]
            }

            # nuScenes possibilities are the Cartesian product of these
            dataset_parts: List[Tuple[str, ...]] = [
                ("test",),
                ("boston", "singapore"),
            ]

            warnings.warn("Beware, nusc_test has no annotations!")
        elif env_name == "nusc_mini":
            nusc_scene_splits: Dict[str, List[str]] = {
                k: all_scene_splits[k] for k in ["mini_train", "mini_val"]
            }

            # nuScenes possibilities are the Cartesian product of these
            dataset_parts: List[Tuple[str, ...]] = [
                ("mini_train", "mini_val"),
                ("boston", "singapore"),
            ]

        # Inverting the dict from above, associating every scene with its data split.
        nusc_scene_split_map: Dict[str, str] = {
            v_elem: k for k, v in nusc_scene_splits.items() for v_elem in v
        }

        return EnvMetadata(
            name=env_name,
            data_dir=data_dir,
            dt=nusc_utils.NUSC_DT,
            parts=dataset_parts,
            scene_split_map=nusc_scene_split_map,
        )

    def load_dataset_obj(self, verbose: bool = False) -> None:
        if verbose:
            print(f"Loading {self.name} dataset...", flush=True)

        if self.name == "nusc_mini":
            version_str = "v1.0-mini"
        elif self.name == "nusc_trainval":
            version_str = "v1.0-trainval"
        elif self.name == "nusc_test":
            version_str = "v1.0-test"

        self.dataset_obj = NuScenes(
            version=version_str, dataroot=self.metadata.data_dir
        )

    def _get_matching_scenes_from_obj(
        self,
        scene_tag: SceneTag,
        scene_desc_contains: Optional[List[str]],
        env_cache: EnvCache,
    ) -> List[SceneMetadata]:
        all_scenes_list: List[NuscSceneRecord] = list()

        scenes_list: List[SceneMetadata] = list()
        for idx, scene_record in enumerate(self.dataset_obj.scene):
            scene_name: str = scene_record["name"]
            scene_desc: str = scene_record["description"].lower()
            scene_location: str = self.dataset_obj.get(
                "log", scene_record["log_token"]
            )["location"]
            scene_split: str = self.metadata.scene_split_map[scene_name]
            scene_length: int = scene_record["nbr_samples"]

            # Saving all scene records for later caching.
            all_scenes_list.append(
                NuscSceneRecord(
                    scene_name, scene_location, scene_length, scene_desc, idx
                )
            )

            if scene_location.split("-")[0] in scene_tag and scene_split in scene_tag:
                if scene_desc_contains is not None and not any(
                    desc_query in scene_desc for desc_query in scene_desc_contains
                ):
                    continue

                scene_metadata = SceneMetadata(
                    env_name=self.metadata.name,
                    name=scene_name,
                    dt=self.metadata.dt,
                    raw_data_idx=idx,
                )
                scenes_list.append(scene_metadata)

        self.cache_all_scenes_list(env_cache, all_scenes_list)
        return scenes_list

    def _get_matching_scenes_from_cache(
        self,
        scene_tag: SceneTag,
        scene_desc_contains: Optional[List[str]],
        env_cache: EnvCache,
    ) -> List[Scene]:
        all_scenes_list: List[NuscSceneRecord] = env_cache.load_env_scenes_list(
            self.name
        )

        scenes_list: List[SceneMetadata] = list()
        for scene_record in all_scenes_list:
            (
                scene_name,
                scene_location,
                scene_length,
                scene_desc,
                data_idx,
            ) = scene_record
            scene_split: str = self.metadata.scene_split_map[scene_name]

            if scene_location.split("-")[0] in scene_tag and scene_split in scene_tag:
                if scene_desc_contains is not None and not any(
                    desc_query in scene_desc for desc_query in scene_desc_contains
                ):
                    continue

                scene_metadata = Scene(
                    self.metadata,
                    scene_name,
                    scene_location,
                    scene_split,
                    scene_length,
                    data_idx,
                    None,  # This isn't used if everything is already cached.
                    scene_desc,
                )
                scenes_list.append(scene_metadata)

        return scenes_list

    def get_scene(self, scene_info: SceneMetadata) -> Scene:
        _, _, _, data_idx = scene_info

        scene_record = self.dataset_obj.scene[data_idx]
        scene_name: str = scene_record["name"]
        scene_desc: str = scene_record["description"].lower()
        scene_location: str = self.dataset_obj.get("log", scene_record["log_token"])[
            "location"
        ]
        scene_split: str = self.metadata.scene_split_map[scene_name]
        scene_length: int = scene_record["nbr_samples"]

        return Scene(
            self.metadata,
            scene_name,
            scene_location,
            scene_split,
            scene_length,
            data_idx,
            scene_record,
            scene_desc,
        )

    def get_agent_info(
        self, scene: Scene, cache_path: Path, cache_class: Type[SceneCache]
    ) -> Tuple[List[AgentMetadata], List[List[AgentMetadata]]]:
        ego_agent_info: AgentMetadata = AgentMetadata(
            name="ego",
            agent_type=AgentType.VEHICLE,
            first_timestep=0,
            last_timestep=scene.length_timesteps - 1,
            extent=FixedExtent(length=4.084, width=1.730, height=1.562),
        )

        agent_presence: List[List[AgentMetadata]] = [
            [ego_agent_info] for _ in range(scene.length_timesteps)
        ]

        agent_data_list: List[pd.DataFrame] = list()
        existing_agents: Dict[str, AgentMetadata] = dict()

        all_frames: List[Dict[str, Union[str, int]]] = list(
            nusc_utils.frame_iterator(self.dataset_obj, scene)
        )
        frame_idx_dict: Dict[str, int] = {
            frame_dict["token"]: idx for idx, frame_dict in enumerate(all_frames)
        }
        for frame_idx, frame_info in enumerate(all_frames):
            for agent_info in nusc_utils.agent_iterator(self.dataset_obj, frame_info):
                if agent_info["instance_token"] in existing_agents:
                    continue

                if not agent_info["next"]:
                    # There are some agents with only a single detection to them, we don't care about these.
                    continue

                agent: Agent = nusc_utils.agg_agent_data(
                    self.dataset_obj, agent_info, frame_idx, frame_idx_dict
                )

                for scene_ts in range(
                    agent.metadata.first_timestep, agent.metadata.last_timestep + 1
                ):
                    agent_presence[scene_ts].append(agent.metadata)

                existing_agents[agent.name] = agent.metadata

                agent_data_list.append(agent.data)

        ego_agent: Agent = nusc_utils.agg_ego_data(self.dataset_obj, scene)
        agent_data_list.append(ego_agent.data)

        agent_list: List[AgentMetadata] = [ego_agent_info] + list(
            existing_agents.values()
        )

        cache_class.save_agent_data(pd.concat(agent_data_list), cache_path, scene)

        return agent_list, agent_presence

    def extract_lane_and_edges(
        self, nusc_map: NuScenesMap, lane_record
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Getting the bounding polygon vertices.
        lane_polygon_obj = nusc_map.get("polygon", lane_record["polygon_token"])
        polygon_nodes = [
            nusc_map.get("node", node_token)
            for node_token in lane_polygon_obj["exterior_node_tokens"]
        ]
        polygon_pts: np.ndarray = np.array(
            [(node["x"], node["y"]) for node in polygon_nodes]
        )

        # Getting the lane center's points.
        curr_lane = nusc_map.arcline_path_3.get(lane_record["token"], [])
        lane_midline: np.ndarray = np.array(
            arcline_path_utils.discretize_lane(curr_lane, resolution_meters=0.5)
        )[:, :2]

        # For some reason, nuScenes duplicates a few entries
        # (likely how they're building their arcline representation).
        # We delete those duplicate entries here.
        duplicate_check: np.ndarray = np.where(
            np.linalg.norm(np.diff(lane_midline, axis=0, prepend=0), axis=1) < 1e-10
        )[0]
        if duplicate_check.size > 0:
            lane_midline = np.delete(lane_midline, duplicate_check, axis=0)

        # Computing the closest lane center point to each bounding polygon vertex.
        closest_midlane_pt: np.ndarray = np.argmin(
            cdist(polygon_pts, lane_midline), axis=1
        )
        # Computing the local direction of the lane at each lane center point.
        direction_vectors: np.ndarray = np.diff(
            lane_midline,
            axis=0,
            prepend=lane_midline[[0]] - (lane_midline[[1]] - lane_midline[[0]]),
        )

        # Selecting the direction vectors at the closest lane center point per polygon vertex.
        local_dir_vecs: np.ndarray = direction_vectors[closest_midlane_pt]
        # Calculating the vectors from the the closest lane center point per polygon vertex to the polygon vertex.
        origin_to_polygon_vecs: np.ndarray = (
            polygon_pts - lane_midline[closest_midlane_pt]
        )

        # Computing the perpendicular dot product.
        # See https://www.xarg.org/book/linear-algebra/2d-perp-product/
        # If perp_dot_product < 0, then the associated polygon vertex is
        # on the right edge of the lane.
        perp_dot_product: np.ndarray = (
            local_dir_vecs[:, 0] * origin_to_polygon_vecs[:, 1]
            - local_dir_vecs[:, 1] * origin_to_polygon_vecs[:, 0]
        )

        # Determining which indices are on the right of the lane center.
        on_right: np.ndarray = perp_dot_product < 0
        # Determining the boundary between the left/right polygon vertices
        # (they will be together in blocks due to the ordering of the polygon vertices).
        idx_changes: int = np.where(np.roll(on_right, 1) < on_right)[0].item()

        if idx_changes > 0:
            # If the block of left/right points spreads across the bounds of the array,
            # roll it until the boundary between left/right points is at index 0.
            # This is important so that the following index selection orders points
            # without jumps.
            polygon_pts = np.roll(polygon_pts, shift=-idx_changes, axis=0)
            on_right = np.roll(on_right, shift=-idx_changes)

        left_pts: np.ndarray = polygon_pts[~on_right]
        right_pts: np.ndarray = polygon_pts[on_right]

        # Final ordering check, ensuring that the beginning of left_pts/right_pts
        # matches the beginning of the lane.
        left_order_correct: bool = np.linalg.norm(
            left_pts[0] - lane_midline[0]
        ) < np.linalg.norm(left_pts[0] - lane_midline[-1])
        right_order_correct: bool = np.linalg.norm(
            right_pts[0] - lane_midline[0]
        ) < np.linalg.norm(right_pts[0] - lane_midline[-1])

        # Reversing left_pts/right_pts in case their first index is
        # at the end of the lane.
        if not left_order_correct:
            left_pts = left_pts[::-1]
        if not right_order_correct:
            right_pts = right_pts[::-1]

        # Ensuring that left and right have the same number of points.
        # This is necessary, not for data storage but for later rasterization.
        if left_pts.shape[0] < right_pts.shape[0]:
            left_pts = map_utils.interpolate(left_pts, right_pts.shape[0])
        elif right_pts.shape[0] < left_pts.shape[0]:
            right_pts = map_utils.interpolate(right_pts, left_pts.shape[0])

        return (
            lane_midline,
            left_pts,
            right_pts,
        )

    def extract_area(self, nusc_map: NuScenesMap, area_record) -> np.ndarray:
        token_key: str
        if "exterior_node_tokens" in area_record:
            token_key = "exterior_node_tokens"
        elif "node_tokens" in area_record:
            token_key = "node_tokens"

        polygon_nodes = [
            nusc_map.get("node", node_token) for node_token in area_record[token_key]
        ]

        return np.array([(node["x"], node["y"]) for node in polygon_nodes])

    def extract_vectorized(self, nusc_map: NuScenesMap) -> VectorizedMap:
        vec_map = VectorizedMap()

        # Setting the map bounds.
        vec_map.max_pt.x, vec_map.max_pt.y, vec_map.max_pt.z = (
            nusc_map.explorer.canvas_max_x,
            nusc_map.explorer.canvas_max_y,
            0.0,
        )
        vec_map.min_pt.x, vec_map.min_pt.y, vec_map.min_pt.z = (
            nusc_map.explorer.canvas_min_x,
            nusc_map.explorer.canvas_min_y,
            0.0,
        )

        overall_pbar = tqdm(
            total=len(nusc_map.lane)
            + len(nusc_map.drivable_area[0]["polygon_tokens"])
            + len(nusc_map.ped_crossing)
            + len(nusc_map.walkway),
            desc=f"Getting {nusc_map.map_name} Elements",
            position=1,
            leave=False,
        )

        for lane_record in nusc_map.lane:
            center_pts, left_pts, right_pts = self.extract_lane_and_edges(
                nusc_map, lane_record
            )

            lane_record_token: str = lane_record["token"]

            # Adding the element to the map.
            new_element: MapElement = vec_map.elements.add()
            new_element.id = lane_record_token.encode()

            new_lane: RoadLane = new_element.road_lane
            map_utils.populate_lane_polylines(new_lane, center_pts, left_pts, right_pts)

            new_lane.entry_lanes.extend(
                lane_id.encode()
                for lane_id in nusc_map.get_incoming_lane_ids(lane_record_token)
            )
            new_lane.exit_lanes.extend(
                lane_id.encode()
                for lane_id in nusc_map.get_outgoing_lane_ids(lane_record_token)
            )

            # new_lane.adjacent_lanes_left.append(
            #     l5_lane.adjacent_lane_change_left.id
            # )
            # new_lane.adjacent_lanes_right.append(
            #     l5_lane.adjacent_lane_change_right.id
            # )

            overall_pbar.update()

        for polygon_token in nusc_map.drivable_area[0]["polygon_tokens"]:
            polygon_record = nusc_map.get("polygon", polygon_token)
            polygon_pts = self.extract_area(nusc_map, polygon_record)

            # Adding the element to the map.
            new_element: MapElement = vec_map.elements.add()
            new_element.id = lane_record["token"].encode()

            new_area: RoadArea = new_element.road_area
            map_utils.populate_polygon(new_area.exterior_polygon, polygon_pts)

            for hole in polygon_record["holes"]:
                polygon_pts = self.extract_area(nusc_map, hole)
                new_hole: Polyline = new_area.interior_holes.add()
                map_utils.populate_polygon(new_hole, polygon_pts)

            overall_pbar.update()

        for ped_area_record in nusc_map.ped_crossing:
            polygon_pts = self.extract_area(nusc_map, ped_area_record)

            # Adding the element to the map.
            new_element: MapElement = vec_map.elements.add()
            new_element.id = ped_area_record["token"].encode()

            new_crosswalk: PedCrosswalk = new_element.ped_crosswalk
            map_utils.populate_polygon(new_crosswalk.polygon, polygon_pts)

            overall_pbar.update()

        for ped_area_record in nusc_map.walkway:
            polygon_pts = self.extract_area(nusc_map, ped_area_record)

            # Adding the element to the map.
            new_element: MapElement = vec_map.elements.add()
            new_element.id = ped_area_record["token"].encode()

            new_walkway: PedWalkway = new_element.ped_walkway
            map_utils.populate_polygon(new_walkway.polygon, polygon_pts)

            overall_pbar.update()

        overall_pbar.close()

        return vec_map

    def cache_map(
        self,
        map_name: str,
        cache_path: Path,
        map_cache_class: Type[SceneCache],
        map_params: Dict[str, Any],
    ) -> None:
        resolution: float = map_params["px_per_m"]

        nusc_map: NuScenesMap = NuScenesMap(
            dataroot=self.metadata.data_dir, map_name=map_name
        )

        if map_params.get("original_format", False):
            warnings.warn(
                "Using a dataset's original map format is deprecated, and will be removed in the next version of trajdata!",
                FutureWarning,
            )

            width_m, height_m = nusc_map.canvas_edge
            height_px, width_px = round(height_m * resolution), round(
                width_m * resolution
            )

            def layer_fn(layer_name: str) -> np.ndarray:
                # Getting rid of the channels dim by accessing index [0]
                return nusc_map.get_map_mask(
                    patch_box=None,
                    patch_angle=0,
                    layer_names=[layer_name],
                    canvas_size=(height_px, width_px),
                )[0].astype(np.bool)

            map_from_world: np.ndarray = np.array(
                [[resolution, 0.0, 0.0], [0.0, resolution, 0.0], [0.0, 0.0, 1.0]]
            )

            layer_names: List[str] = [
                "lane",
                "road_segment",
                "drivable_area",
                "road_divider",
                "lane_divider",
                "ped_crossing",
                "walkway",
            ]
            map_info: RasterizedMapMetadata = RasterizedMapMetadata(
                name=map_name,
                shape=(len(layer_names), height_px, width_px),
                layers=layer_names,
                layer_rgb_groups=([0, 1, 2], [3, 4], [5, 6]),
                resolution=resolution,
                map_from_world=map_from_world,
            )

            map_cache_class.cache_map_layers(
                cache_path, VectorizedMap(), map_info, layer_fn, self.name
            )
        else:
            vectorized_map: VectorizedMap = self.extract_vectorized(nusc_map)

            pbar_kwargs = {"position": 2, "leave": False}
            map_data, map_from_world = map_utils.rasterize_map(
                vectorized_map, resolution, **pbar_kwargs
            )

            rasterized_map_info: RasterizedMapMetadata = RasterizedMapMetadata(
                name=map_name,
                shape=map_data.shape,
                layers=["drivable_area", "lane_divider", "ped_area"],
                layer_rgb_groups=([0], [1], [2]),
                resolution=resolution,
                map_from_world=map_from_world,
            )

            rasterized_map_obj: RasterizedMap = RasterizedMap(
                rasterized_map_info, map_data
            )

            map_cache_class.cache_map(
                cache_path, vectorized_map, rasterized_map_obj, self.name
            )

    def cache_maps(
        self,
        cache_path: Path,
        map_cache_class: Type[SceneCache],
        map_params: Dict[str, Any],
    ) -> None:
        """
        Stores rasterized maps to disk for later retrieval.

        Below are the map origins (south western corner, in [lat, lon]) for each of
        the 4 maps in nuScenes:
            boston-seaport: [42.336849169438615, -71.05785369873047]
            singapore-onenorth: [1.2882100868743724, 103.78475189208984]
            singapore-hollandvillage: [1.2993652317780957, 103.78217697143555]
            singapore-queenstown: [1.2782562240223188, 103.76741409301758]

        The dimensions of the maps are as follows ([width, height] in meters). They
        can also be found in nusc_utils.py
            singapore-onenorth:       [1585.6, 2025.0]
            singapore-hollandvillage: [2808.3, 2922.9]
            singapore-queenstown:     [3228.6, 3687.1]
            boston-seaport:           [2979.5, 2118.1]
        The rasterized semantic maps published with nuScenes v1.0 have a scale of 10px/m,
        hence the above numbers are the image dimensions divided by 10.

        nuScenes uses the same WGS 84 Web Mercator (EPSG:3857) projection as Google Maps/Earth.
        """
        for map_name in tqdm(
            locations,
            desc=f"Caching {self.name} Maps at {map_params['px_per_m']:.2f} px/m",
            position=0,
        ):
            self.cache_map(map_name, cache_path, map_cache_class, map_params)
