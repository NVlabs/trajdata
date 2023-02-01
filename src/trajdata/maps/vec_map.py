from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trajdata.maps.map_kdtree import MapElementKDTree, LaneCenterKDTree

from collections import defaultdict
from dataclasses import dataclass, field
from math import ceil
from typing import (
    DefaultDict,
    Dict,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Union,
    overload,
)

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from tqdm import tqdm

import trajdata.proto.vectorized_map_pb2 as map_proto
from trajdata.maps.map_kdtree import LaneCenterKDTree
from trajdata.maps.traffic_light_status import TrafficLightStatus
from trajdata.maps.vec_map_elements import (
    MapElement,
    MapElementType,
    PedCrosswalk,
    PedWalkway,
    Polyline,
    RoadArea,
    RoadLane,
)
from trajdata.utils import map_utils, raster_utils


@dataclass(repr=False)
class VectorMap:
    map_id: str
    extent: Optional[
        np.ndarray
    ] = None  # extent is [min_x, min_y, min_z, max_x, max_y, max_z]
    elements: DefaultDict[MapElementType, Dict[str, MapElement]] = field(
        default_factory=lambda: defaultdict(dict)
    )
    search_kdtrees: Optional[Dict[MapElementType, MapElementKDTree]] = None
    traffic_light_status: Optional[Dict[Tuple[int, int], TrafficLightStatus]] = None

    def __post_init__(self) -> None:
        self.env_name, self.map_name = self.map_id.split(":")

        self.lanes: Optional[List[RoadLane]] = None
        if MapElementType.ROAD_LANE in self.elements:
            self.lanes = list(self.elements[MapElementType.ROAD_LANE].values())

    def add_map_element(self, map_elem: MapElement) -> None:
        self.elements[map_elem.elem_type][map_elem.id] = map_elem

    def compute_search_indices(self) -> None:
        self.search_kdtrees = {MapElementType.ROAD_LANE: LaneCenterKDTree(self)}

    def iter_elems(self) -> Iterator[MapElement]:
        for elems_dict in self.elements.values():
            for elem in elems_dict.values():
                yield elem

    def get_road_lane(self, lane_id: str) -> RoadLane:
        return self.elements[MapElementType.ROAD_LANE][lane_id]

    def __len__(self) -> int:
        return sum(len(elems_dict) for elems_dict in self.elements.values())

    def _write_road_lanes(
        self, vectorized_map: map_proto.VectorizedMap, shifted_origin: np.ndarray
    ) -> None:
        road_lane: RoadLane
        for elem_id, road_lane in self.elements[MapElementType.ROAD_LANE].items():
            new_element: map_proto.MapElement = vectorized_map.elements.add()
            new_element.id = elem_id.encode()

            new_lane: map_proto.RoadLane = new_element.road_lane
            map_utils.populate_lane_polylines(new_lane, road_lane, shifted_origin)

            new_lane.entry_lanes.extend(
                [lane_id.encode() for lane_id in road_lane.prev_lanes]
            )
            new_lane.exit_lanes.extend(
                [lane_id.encode() for lane_id in road_lane.next_lanes]
            )

            new_lane.adjacent_lanes_left.extend(
                [lane_id.encode() for lane_id in road_lane.adj_lanes_left]
            )
            new_lane.adjacent_lanes_right.extend(
                [lane_id.encode() for lane_id in road_lane.adj_lanes_right]
            )

    def _write_road_areas(
        self, vectorized_map: map_proto.VectorizedMap, shifted_origin: np.ndarray
    ) -> None:
        road_area: RoadArea
        for elem_id, road_area in self.elements[MapElementType.ROAD_AREA].items():
            new_element: map_proto.MapElement = vectorized_map.elements.add()
            new_element.id = elem_id.encode()

            new_area: map_proto.RoadArea = new_element.road_area
            map_utils.populate_polygon(
                new_area.exterior_polygon,
                road_area.exterior_polygon.xyz,
                shifted_origin,
            )

            hole: Polyline
            for hole in road_area.interior_holes:
                new_hole: map_proto.Polyline = new_area.interior_holes.add()
                map_utils.populate_polygon(
                    new_hole,
                    hole.xyz,
                    shifted_origin,
                )

    def _write_ped_crosswalks(
        self, vectorized_map: map_proto.VectorizedMap, shifted_origin: np.ndarray
    ) -> None:
        ped_crosswalk: PedCrosswalk
        for elem_id, ped_crosswalk in self.elements[
            MapElementType.PED_CROSSWALK
        ].items():
            new_element: map_proto.MapElement = vectorized_map.elements.add()
            new_element.id = elem_id.encode()

            new_crosswalk: map_proto.PedCrosswalk = new_element.ped_crosswalk
            map_utils.populate_polygon(
                new_crosswalk.polygon,
                ped_crosswalk.polygon.xyz,
                shifted_origin,
            )

    def _write_ped_walkways(
        self, vectorized_map: map_proto.VectorizedMap, shifted_origin: np.ndarray
    ) -> None:
        ped_walkway: PedWalkway
        for elem_id, ped_walkway in self.elements[MapElementType.PED_WALKWAY].items():
            new_element: map_proto.MapElement = vectorized_map.elements.add()
            new_element.id = elem_id.encode()

            new_walkway: map_proto.PedWalkway = new_element.ped_walkway
            map_utils.populate_polygon(
                new_walkway.polygon,
                ped_walkway.polygon.xyz,
                shifted_origin,
            )

    def to_proto(self) -> map_proto.VectorizedMap:
        output_map = map_proto.VectorizedMap()
        output_map.name = self.map_id

        (
            output_map.min_pt.x,
            output_map.min_pt.y,
            output_map.min_pt.z,
            output_map.max_pt.x,
            output_map.max_pt.y,
            output_map.max_pt.z,
        ) = self.extent

        shifted_origin: np.ndarray = self.extent[:3]
        (
            output_map.shifted_origin.x,
            output_map.shifted_origin.y,
            output_map.shifted_origin.z,
        ) = shifted_origin

        # Populating the elements in the vectorized map protobuf.
        self._write_road_lanes(output_map, shifted_origin)
        self._write_road_areas(output_map, shifted_origin)
        self._write_ped_crosswalks(output_map, shifted_origin)
        self._write_ped_walkways(output_map, shifted_origin)

        return output_map

    @classmethod
    def from_proto(cls, vec_map: map_proto.VectorizedMap, **kwargs):
        # Options for which map elements to include.
        incl_road_lanes: bool = kwargs.get("incl_road_lanes", True)
        incl_road_areas: bool = kwargs.get("incl_road_areas", False)
        incl_ped_crosswalks: bool = kwargs.get("incl_ped_crosswalks", False)
        incl_ped_walkways: bool = kwargs.get("incl_ped_walkways", False)

        # Add any map offset in case the map origin was shifted for storage efficiency.
        shifted_origin: np.ndarray = np.array(
            [
                vec_map.shifted_origin.x,
                vec_map.shifted_origin.y,
                vec_map.shifted_origin.z,
                0.0,  # Some polylines also have heading so we're adding
                # this (zero) coordinate to account for that.
            ]
        )

        map_elem_dict: Dict[str, Dict[str, MapElement]] = defaultdict(dict)

        map_elem: MapElement
        for map_elem in vec_map.elements:
            elem_id: str = map_elem.id.decode()
            if incl_road_lanes and map_elem.HasField("road_lane"):
                road_lane_obj: map_proto.RoadLane = map_elem.road_lane

                center_pl: Polyline = Polyline(
                    map_utils.proto_to_np(road_lane_obj.center) + shifted_origin
                )

                # We do not care for the heading of the left and right edges
                # (only the center matters).
                left_pl: Optional[Polyline] = None
                if road_lane_obj.HasField("left_boundary"):
                    left_pl = Polyline(
                        map_utils.proto_to_np(
                            road_lane_obj.left_boundary, incl_heading=False
                        )
                        + shifted_origin[:3]
                    )

                right_pl: Optional[Polyline] = None
                if road_lane_obj.HasField("right_boundary"):
                    right_pl = Polyline(
                        map_utils.proto_to_np(
                            road_lane_obj.right_boundary, incl_heading=False
                        )
                        + shifted_origin[:3]
                    )

                adj_lanes_left: Set[str] = set(
                    [iden.decode() for iden in road_lane_obj.adjacent_lanes_left]
                )
                adj_lanes_right: Set[str] = set(
                    [iden.decode() for iden in road_lane_obj.adjacent_lanes_right]
                )

                next_lanes: Set[str] = set(
                    [iden.decode() for iden in road_lane_obj.exit_lanes]
                )
                prev_lanes: Set[str] = set(
                    [iden.decode() for iden in road_lane_obj.entry_lanes]
                )

                # Double-using the connectivity attributes for lane IDs now (will
                # replace them with Lane objects after all Lane objects have been created).
                curr_lane = RoadLane(
                    elem_id,
                    center_pl,
                    left_pl,
                    right_pl,
                    adj_lanes_left,
                    adj_lanes_right,
                    next_lanes,
                    prev_lanes,
                )
                map_elem_dict[MapElementType.ROAD_LANE][elem_id] = curr_lane

            elif incl_road_areas and map_elem.HasField("road_area"):
                road_area_obj: map_proto.RoadArea = map_elem.road_area

                exterior: Polyline = Polyline(
                    map_utils.proto_to_np(
                        road_area_obj.exterior_polygon, incl_heading=False
                    )
                    + shifted_origin[:3]
                )

                interior_holes: List[Polyline] = list()
                interior_hole: map_proto.Polyline
                for interior_hole in road_area_obj.interior_holes:
                    interior_holes.append(
                        Polyline(
                            map_utils.proto_to_np(interior_hole, incl_heading=False)
                            + shifted_origin[:3]
                        )
                    )

                curr_area = RoadArea(elem_id, exterior, interior_holes)
                map_elem_dict[MapElementType.ROAD_AREA][elem_id] = curr_area

            elif incl_ped_crosswalks and map_elem.HasField("ped_crosswalk"):
                ped_crosswalk_obj: map_proto.PedCrosswalk = map_elem.ped_crosswalk

                polygon_vertices: Polyline = Polyline(
                    map_utils.proto_to_np(ped_crosswalk_obj.polygon, incl_heading=False)
                    + shifted_origin[:3]
                )

                curr_area = PedCrosswalk(elem_id, polygon_vertices)
                map_elem_dict[MapElementType.PED_CROSSWALK][elem_id] = curr_area

            elif incl_ped_walkways and map_elem.HasField("ped_walkway"):
                ped_walkway_obj: map_proto.PedCrosswalk = map_elem.ped_walkway

                polygon_vertices: Polyline = Polyline(
                    map_utils.proto_to_np(ped_walkway_obj.polygon, incl_heading=False)
                    + shifted_origin[:3]
                )

                curr_area = PedWalkway(elem_id, polygon_vertices)
                map_elem_dict[MapElementType.PED_WALKWAY][elem_id] = curr_area

        return cls(
            map_id=vec_map.name,
            extent=np.array(
                [
                    vec_map.min_pt.x,
                    vec_map.min_pt.y,
                    vec_map.min_pt.z,
                    vec_map.max_pt.x,
                    vec_map.max_pt.y,
                    vec_map.max_pt.z,
                ]
            ),
            elements=map_elem_dict,
            search_kdtrees=None,
            traffic_light_status=None,
        )

    def associate_scene_data(
        self, traffic_light_status_dict: Dict[Tuple[int, int], TrafficLightStatus]
    ) -> None:
        """Associates vector map with scene-specific data like traffic light information"""
        self.traffic_light_status = traffic_light_status_dict

    def get_current_lane(
        self,
        xyzh: np.ndarray,
        max_dist: float = 2.0,
        max_heading_error: float = np.pi / 8,
    ) -> List[RoadLane]:
        """
        Args:
            xyzh (np.ndarray): 3d position and heading of agent in world coordinates

        Returns:
            List[RoadLane]: List of possible road lanes that agent could be on
        """
        lane_kdtree: LaneCenterKDTree = self.search_kdtrees[MapElementType.ROAD_LANE]
        return [
            self.lanes[idx]
            for idx in lane_kdtree.current_lane_inds(xyzh, max_dist, max_heading_error)
        ]

    def get_closest_lane(self, xyz: np.ndarray) -> RoadLane:
        lane_kdtree: LaneCenterKDTree = self.search_kdtrees[MapElementType.ROAD_LANE]
        return self.lanes[lane_kdtree.closest_polyline_ind(xyz)]

    def get_closest_unique_lanes(self, xyz_vec: np.ndarray) -> List[RoadLane]:
        assert xyz_vec.ndim == 2  # xyz_vec is assumed to be (*, 3)
        lane_kdtree: LaneCenterKDTree = self.search_kdtrees[MapElementType.ROAD_LANE]
        closest_inds = lane_kdtree.closest_polyline_ind(xyz_vec)
        unique_inds = np.unique(closest_inds)
        return [self.lanes[ind] for ind in unique_inds]

    def get_lanes_within(self, xyz: np.ndarray, dist: float) -> List[RoadLane]:
        lane_kdtree: LaneCenterKDTree = self.search_kdtrees[MapElementType.ROAD_LANE]
        return [
            self.lanes[idx] for idx in lane_kdtree.polyline_inds_in_range(xyz, dist)
        ]

    def get_traffic_light_status(
        self, lane_id: str, scene_ts: int
    ) -> TrafficLightStatus:
        return (
            self.traffic_light_status.get(
                (int(lane_id), scene_ts), TrafficLightStatus.NO_DATA
            )
            if self.traffic_light_status is not None
            else TrafficLightStatus.NO_DATA
        )

    def rasterize(
        self, resolution: float = 2, **kwargs
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Renders this vector map at the specified resolution.

        Args:
            resolution (float): The rasterized image's resolution in pixels per meter.

        Returns:
            np.ndarray: The rasterized RGB image.
        """
        return_tf_mat: bool = kwargs.get("return_tf_mat", False)
        incl_centerlines: bool = kwargs.get("incl_centerlines", True)
        incl_lane_edges: bool = kwargs.get("incl_lane_edges", True)
        incl_lane_area: bool = kwargs.get("incl_lane_area", True)

        scene_ts: Optional[int] = kwargs.get("scene_ts", None)

        # (255, 102, 99) also looks nice.
        center_color: Tuple[int, int, int] = kwargs.get("center_color", (129, 51, 255))
        # (86, 203, 249) also looks nice.
        edge_color: Tuple[int, int, int] = kwargs.get("edge_color", (118, 185, 0))
        # (191, 215, 234) also looks nice.
        area_color: Tuple[int, int, int] = kwargs.get("area_color", (214, 232, 181))

        min_x, min_y, _, max_x, max_y, _ = self.extent

        world_center_m: Tuple[float, float] = (
            (max_x + min_x) / 2,
            (max_y + min_y) / 2,
        )

        raster_size_x: int = ceil((max_x - min_x) * resolution)
        raster_size_y: int = ceil((max_y - min_y) * resolution)

        raster_from_local: np.ndarray = np.array(
            [
                [resolution, 0, raster_size_x / 2],
                [0, resolution, raster_size_y / 2],
                [0, 0, 1],
            ]
        )

        # Compute pose from its position and rotation.
        pose_from_world: np.ndarray = np.array(
            [
                [1, 0, -world_center_m[0]],
                [0, 1, -world_center_m[1]],
                [0, 0, 1],
            ]
        )

        raster_from_world: np.ndarray = raster_from_local @ pose_from_world

        map_img: np.ndarray = np.zeros(
            shape=(raster_size_y, raster_size_x, 3), dtype=np.uint8
        )

        lane_edges: List[np.ndarray] = list()
        centerlines: List[np.ndarray] = list()
        lane: RoadLane
        for lane in tqdm(
            self.elements[MapElementType.ROAD_LANE].values(),
            desc=f"Rasterizing Map at {resolution:.2f} px/m",
            leave=False,
        ):
            centerlines.append(
                raster_utils.world_to_subpixel(
                    lane.center.points[:, :2], raster_from_world
                )
            )
            if lane.left_edge is not None and lane.right_edge is not None:
                left_pts: np.ndarray = lane.left_edge.points[:, :2]
                right_pts: np.ndarray = lane.right_edge.points[:, :2]

                lane_edges += [
                    raster_utils.world_to_subpixel(left_pts, raster_from_world),
                    raster_utils.world_to_subpixel(right_pts, raster_from_world),
                ]

                lane_color = area_color
                status = self.get_traffic_light_status(lane.id, scene_ts)
                if status == TrafficLightStatus.GREEN:
                    lane_color = [0, 200, 0]
                elif status == TrafficLightStatus.RED:
                    lane_color = [200, 0, 0]
                elif status == TrafficLightStatus.UNKNOWN:
                    lane_color = [150, 150, 0]

                # Drawing lane areas. Need to do per loop because doing it all at once can
                # create lots of wonky holes in the image.
                # See https://stackoverflow.com/questions/69768620/cv2-fillpoly-failing-for-intersecting-polygons
                if incl_lane_area:
                    lane_area: np.ndarray = np.concatenate(
                        [left_pts, right_pts[::-1]], axis=0
                    )
                    raster_utils.rasterize_world_polygon(
                        lane_area,
                        map_img,
                        raster_from_world,
                        color=lane_color,
                    )

        # Drawing all lane edge lines at the same time.
        if incl_lane_edges:
            raster_utils.cv2_draw_polylines(lane_edges, map_img, color=edge_color)

        # Drawing centerlines last (on top of everything else).
        if incl_centerlines:
            raster_utils.cv2_draw_polylines(centerlines, map_img, color=center_color)

        if return_tf_mat:
            return map_img.astype(float) / 255, raster_from_world
        else:
            return map_img.astype(float) / 255

    @overload
    def visualize_lane_graph(
        self,
        origin_lane: RoadLane,
        num_hops: int,
        **kwargs,
    ) -> Axes:
        ...

    @overload
    def visualize_lane_graph(self, origin_lane: str, num_hops: int, **kwargs) -> Axes:
        ...

    @overload
    def visualize_lane_graph(self, origin_lane: int, num_hops: int, **kwargs) -> Axes:
        ...

    def visualize_lane_graph(
        self, origin_lane: Union[RoadLane, str, int], num_hops: int, **kwargs
    ) -> Axes:
        ax = kwargs.get("ax", None)
        if ax is None:
            fig, ax = plt.subplots()

        origin: str
        if isinstance(origin_lane, RoadLane):
            origin = origin_lane.id
        elif isinstance(origin_lane, str):
            origin = origin_lane
        elif isinstance(origin_lane, int):
            origin = self.lanes[origin_lane].id

        viridis = mpl.colormaps[kwargs.get("cmap", "rainbow")].resampled(num_hops + 1)

        already_seen: Set[str] = set()
        lanes_to_plot: List[Tuple[str, int]] = [(origin, 0)]

        if kwargs.get("legend", True):
            ax.scatter([], [], label=f"Lane Endpoints", color="k")
            ax.plot([], [], label=f"Origin Lane ({origin})", color=viridis(0))
            for h in range(1, num_hops + 1):
                ax.plot(
                    [],
                    [],
                    label=f"{h} Lane{'s' if h > 1 else ''} Away",
                    color=viridis(h),
                )

        raster_from_world = kwargs.get("raster_from_world", None)
        while len(lanes_to_plot) > 0:
            lane_id, curr_hops = lanes_to_plot.pop(0)
            already_seen.add(lane_id)
            lane: RoadLane = self.get_road_lane(lane_id)

            center: np.ndarray = lane.center.points[..., :2]
            first_pt_heading: float = lane.center.points[0, -1]
            mdpt: np.ndarray = lane.center.midpoint[..., :2]

            if raster_from_world is not None:
                center = map_utils.transform_points(center, raster_from_world)
                mdpt = map_utils.transform_points(mdpt[None, :], raster_from_world)[0]

            ax.plot(center[:, 0], center[:, 1], color=viridis(curr_hops))
            ax.scatter(center[[0, -1], 0], center[[0, -1], 1], color=viridis(curr_hops))
            ax.quiver(
                [center[0, 0]],
                [center[0, 1]],
                [np.cos(first_pt_heading)],
                [np.sin(first_pt_heading)],
                color=viridis(curr_hops),
            )
            ax.text(mdpt[0], mdpt[1], s=lane_id)

            if curr_hops < num_hops:
                lanes_to_plot += [
                    (l, curr_hops + 1)
                    for l in lane.reachable_lanes
                    if l not in already_seen
                ]

        if kwargs.get("legend", True):
            ax.legend(loc="best", frameon=True)

        return ax
