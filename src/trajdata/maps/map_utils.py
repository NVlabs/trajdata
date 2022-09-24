from math import ceil
from typing import Any, Dict, Final, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from trajdata.proto.vectorized_map_pb2 import (
    MapElement,
    Polyline,
    RoadLane,
    VectorizedMap,
)

# Sub-pixel drawing precision constants.
# See https://github.com/woven-planet/l5kit/blob/master/l5kit/l5kit/rasterization/semantic_rasterizer.py#L16
CV2_SUB_VALUES = {"shift": 9, "lineType": cv2.LINE_AA}
CV2_SHIFT_VALUE = 2 ** CV2_SUB_VALUES["shift"]

MM_PER_M: Final[float] = 1000


def cv2_subpixel(coords: np.ndarray) -> np.ndarray:
    """
    Cast coordinates to numpy.int but keep fractional part by previously multiplying by 2**CV2_SHIFT
    cv2 calls will use shift to restore original values with higher precision

    Args:
        coords (np.ndarray): XY coords as float

    Returns:
        np.ndarray: XY coords as int for cv2 shift draw
    """
    return (coords * CV2_SHIFT_VALUE).astype(np.int)


def decompress_values(data: np.ndarray) -> np.ndarray:
    # From https://github.com/woven-planet/l5kit/blob/master/l5kit/l5kit/data/proto/road_network.proto#L446
    # The delta for the first point is just its coordinates tuple, i.e. it is a "delta" from
    # the origin. For subsequent points, this field stores the difference between the point's
    # coordinates and the previous point's coordinates. This is for representation efficiency.
    return np.cumsum(data, axis=0, dtype=np.float) / MM_PER_M


def compress_values(data: np.ndarray) -> np.ndarray:
    return (np.diff(data, axis=0, prepend=0.0) * MM_PER_M).astype(np.int32)


def populate_lane_polylines(
    new_lane: RoadLane,
    midlane_pts: np.ndarray,
    left_pts: np.ndarray,
    right_pts: np.ndarray,
) -> None:
    """Fill a Lane object's polyline attributes.
    All points should be in world coordinates.

    Args:
        new_lane (Lane): _description_
        midlane_pts (np.ndarray): _description_
        left_pts (np.ndarray): _description_
        right_pts (np.ndarray): _description_
    """
    compressed_mid_pts: np.ndarray = compress_values(midlane_pts)
    compressed_left_pts: np.ndarray = compress_values(left_pts)
    compressed_right_pts: np.ndarray = compress_values(right_pts)

    new_lane.center.dx_mm.extend(compressed_mid_pts[:, 0].tolist())
    new_lane.center.dy_mm.extend(compressed_mid_pts[:, 1].tolist())

    new_lane.left_boundary.dx_mm.extend(compressed_left_pts[:, 0].tolist())
    new_lane.left_boundary.dy_mm.extend(compressed_left_pts[:, 1].tolist())

    new_lane.right_boundary.dx_mm.extend(compressed_right_pts[:, 0].tolist())
    new_lane.right_boundary.dy_mm.extend(compressed_right_pts[:, 1].tolist())

    if compressed_mid_pts.shape[-1] == 3:
        new_lane.center.dz_mm.extend(compressed_mid_pts[:, 2].tolist())
        new_lane.left_boundary.dz_mm.extend(compressed_left_pts[:, 2].tolist())
        new_lane.right_boundary.dz_mm.extend(compressed_right_pts[:, 2].tolist())


def populate_polygon(
    polygon: Polyline,
    polygon_pts: np.ndarray,
) -> None:
    """Fill a Crosswalk object's polygon attribute.
    All points should be in world coordinates.

    Args:
        new_crosswalk (Lane): _description_
        polygon_pts (np.ndarray): _description_
    """

    compressed_pts: np.ndarray = compress_values(polygon_pts)

    polygon.dx_mm.extend(compressed_pts[:, 0].tolist())
    polygon.dy_mm.extend(compressed_pts[:, 1].tolist())

    if compressed_pts.shape[-1] == 3:
        polygon.dz_mm.extend(compressed_pts[:, 2].tolist())


def proto_to_np(polyline: Polyline) -> np.ndarray:
    dx: np.ndarray = np.asarray(polyline.dx_mm)
    dy: np.ndarray = np.asarray(polyline.dy_mm)

    if len(polyline.dz_mm) > 0:
        dz: np.ndarray = np.asarray(polyline.dz_mm)
        pts: np.ndarray = np.stack([dx, dy, dz], axis=1)
    else:
        pts: np.ndarray = np.stack([dx, dy], axis=1)

    return decompress_values(pts)


def transform_points(points: np.ndarray, transf_mat: np.ndarray):
    n_dim = points.shape[-1]
    return points @ transf_mat[:n_dim, :n_dim] + transf_mat[:n_dim, -1]


def interpolate(pts: np.ndarray, num_pts: int) -> np.ndarray:
    """
    Interpolate points based on cumulative distances from the first one. In particular,
    interpolate using a variable step such that we always get step values.

    Args:
        xyz (np.ndarray): XYZ coords.
        num_pts (int): How many points to interpolate to.

    Returns:
        np.ndarray: The new interpolated coordinates.
    """
    cum_dist = np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=-1))
    cum_dist = np.insert(cum_dist, 0, 0)

    assert num_pts > 1, f"num_pts must be at least 2, but got {num_pts}"
    steps = np.linspace(cum_dist[0], cum_dist[-1], num_pts)

    xyz_inter = np.empty((len(steps), pts.shape[-1]), dtype=pts.dtype)
    xyz_inter[:, 0] = np.interp(steps, xp=cum_dist, fp=pts[:, 0])
    xyz_inter[:, 1] = np.interp(steps, xp=cum_dist, fp=pts[:, 1])
    if pts.shape[-1] == 3:
        xyz_inter[:, 2] = np.interp(steps, xp=cum_dist, fp=pts[:, 2])

    return xyz_inter


def rasterize_map(
    vec_map: VectorizedMap, resolution: float, **pbar_kwargs
) -> np.ndarray:
    """Renders the semantic map at the given resolution.

    Args:
        vec_map (VectorizedMap): _description_
        resolution (float): The rasterized image's resolution in pixels per meter.

    Returns:
        np.ndarray: The rasterized RGB image.
    """
    world_center_m: Tuple[float, float] = (
        (vec_map.max_pt.x + vec_map.min_pt.x) / 2,
        (vec_map.max_pt.y + vec_map.min_pt.y) / 2,
    )

    raster_size_x: int = ceil((vec_map.max_pt.x - vec_map.min_pt.x) * resolution)
    raster_size_y: int = ceil((vec_map.max_pt.y - vec_map.min_pt.y) * resolution)

    raster_from_local: np.ndarray = np.array(
        [
            [resolution, 0, raster_size_x / 2],
            [0, resolution, raster_size_y / 2],
            [0, 0, 1],
        ]
    )

    # Compute pose from its position and rotation
    pose_from_world: np.ndarray = np.array(
        [
            [1, 0, -world_center_m[0]],
            [0, 1, -world_center_m[1]],
            [0, 0, 1],
        ]
    )

    raster_from_world: np.ndarray = raster_from_local @ pose_from_world

    lane_area_img: np.ndarray = np.zeros(
        shape=(raster_size_y, raster_size_x, 3), dtype=np.uint8
    )
    lane_line_img: np.ndarray = np.zeros(
        shape=(raster_size_y, raster_size_x, 3), dtype=np.uint8
    )
    ped_area_img: np.ndarray = np.zeros(
        shape=(raster_size_y, raster_size_x, 3), dtype=np.uint8
    )

    map_elem: MapElement
    for map_elem in tqdm(
        vec_map.elements,
        desc=f"Rasterizing Map at {resolution:.2f} px/m",
        **pbar_kwargs,
    ):
        if map_elem.HasField("road_lane"):
            left_pts: np.ndarray = proto_to_np(map_elem.road_lane.left_boundary)
            right_pts: np.ndarray = proto_to_np(map_elem.road_lane.right_boundary)

            lane_area: np.ndarray = cv2_subpixel(
                transform_points(
                    np.concatenate([left_pts[:, :2], right_pts[::-1, :2]], axis=0),
                    raster_from_world,
                )
            )

            # Need to for-loop because doing it all at once can make holes.
            cv2.fillPoly(
                img=lane_area_img,
                pts=[lane_area],
                color=(255, 0, 0),
                **CV2_SUB_VALUES,
            )

            # Drawing lane lines.
            cv2.polylines(
                img=lane_line_img,
                pts=lane_area.reshape((2, -1, 2)),
                isClosed=False,
                color=(0, 255, 0),
                **CV2_SUB_VALUES,
            )

        elif map_elem.HasField("road_area"):
            xyz_pts: np.ndarray = proto_to_np(map_elem.road_area.exterior_polygon)
            road_area: np.ndarray = cv2_subpixel(
                transform_points(xyz_pts[:, :2], raster_from_world)
            )

            # Drawing general road areas.
            cv2.fillPoly(
                img=lane_area_img,
                pts=[road_area],
                color=(255, 0, 0),
                **CV2_SUB_VALUES,
            )

            for interior_hole in map_elem.road_area.interior_holes:
                xyz_pts: np.ndarray = proto_to_np(interior_hole)
                road_area: np.ndarray = cv2_subpixel(
                    transform_points(xyz_pts[:, :2], raster_from_world)
                )

                # Removing holes.
                cv2.fillPoly(
                    img=lane_area_img,
                    pts=[road_area],
                    color=(0, 0, 0),
                    **CV2_SUB_VALUES,
                )

        elif map_elem.HasField("ped_crosswalk"):
            xyz_pts: np.ndarray = proto_to_np(map_elem.ped_crosswalk.polygon)
            crosswalk_area: np.ndarray = cv2_subpixel(
                transform_points(xyz_pts[:, :2], raster_from_world)
            )

            # Drawing crosswalks.
            cv2.fillPoly(
                img=ped_area_img,
                pts=[crosswalk_area],
                color=(0, 0, 255),
                **CV2_SUB_VALUES,
            )

        elif map_elem.HasField("ped_walkway"):
            xyz_pts: np.ndarray = proto_to_np(map_elem.ped_walkway.polygon)
            walkway_area: np.ndarray = cv2_subpixel(
                transform_points(xyz_pts[:, :2], raster_from_world)
            )

            # Drawing walkways.
            cv2.fillPoly(
                img=ped_area_img,
                pts=[walkway_area],
                color=(0, 0, 255),
                **CV2_SUB_VALUES,
            )

    map_img: np.ndarray = (lane_area_img + lane_line_img + ped_area_img).astype(
        np.float32
    ) / 255
    return map_img.transpose(2, 0, 1), raster_from_world
