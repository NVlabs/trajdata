from collections import defaultdict
from typing import Dict

import cv2
import numpy as np
from l5kit.data.map_api import InterpolationMethod
from l5kit.geometry import transform_points
from l5kit.rasterization.semantic_rasterizer import (
    CV2_SUB_VALUES,
    INTERPOLATION_POINTS,
    RasterEls,
    SemanticRasterizer,
    cv2_subpixel,
)


def indices_in_bounds(
    center: np.ndarray, bounds: np.ndarray, half_extent: float
) -> np.ndarray:
    """
    Get indices of elements for which the bounding box described by bounds intersects the one defined around
    center (square with side 2*half_side)

    Args:
        center (float): XY of the center
        bounds (np.ndarray): array of shape Nx2x2 [[x_min,y_min],[x_max, y_max]]
        half_extent (float): half the side of the bounding box centered around center

    Returns:
        np.ndarray: indices of elements inside radius from center
    """
    return np.arange(bounds.shape[0], dtype=np.long)


class MapSemanticRasterizer(SemanticRasterizer):
    def render_semantic_map(
        self, center_in_world: np.ndarray, raster_from_world: np.ndarray
    ) -> np.ndarray:
        """Renders the semantic map at given x,y coordinates.

        Args:
            center_in_world (np.ndarray): XY of the image center in world ref system
            raster_from_world (np.ndarray):
        Returns:
            np.ndarray: RGB raster

        """
        lane_area_img: np.ndarray = np.zeros(
            shape=(self.raster_size[1], self.raster_size[0], 3), dtype=np.uint8
        )
        lane_line_img: np.ndarray = np.zeros(
            shape=(self.raster_size[1], self.raster_size[0], 3), dtype=np.uint8
        )
        ped_area_img: np.ndarray = np.zeros(
            shape=(self.raster_size[1], self.raster_size[0], 3), dtype=np.uint8
        )

        # filter using half a radius from the center
        raster_radius = float(np.linalg.norm(self.raster_size * self.pixel_size)) / 2

        # get all lanes as interpolation so that we can transform them all together
        lane_indices = indices_in_bounds(
            center_in_world, self.mapAPI.bounds_info["lanes"]["bounds"], raster_radius
        )
        lanes_mask: Dict[str, np.ndarray] = defaultdict(
            lambda: np.zeros(len(lane_indices) * 2, dtype=np.bool)
        )
        lanes_area = np.zeros((len(lane_indices) * 2, INTERPOLATION_POINTS, 2))

        for idx, lane_idx in enumerate(lane_indices):
            lane_idx = self.mapAPI.bounds_info["lanes"]["ids"][lane_idx]

            # interpolate over polyline to always have the same number of points
            lane_coords = self.mapAPI.get_lane_as_interpolation(
                lane_idx, INTERPOLATION_POINTS, InterpolationMethod.INTER_ENSURE_LEN
            )
            lanes_area[idx * 2] = lane_coords["xyz_left"][:, :2]
            lanes_area[idx * 2 + 1] = lane_coords["xyz_right"][::-1, :2]

            lanes_mask[RasterEls.LANE_NOTL.name][idx * 2 : idx * 2 + 2] = True

        if len(lanes_area):
            lanes_area = cv2_subpixel(
                transform_points(lanes_area.reshape((-1, 2)), raster_from_world)
            )

            for lane_area in lanes_area.reshape((-1, INTERPOLATION_POINTS * 2, 2)):
                # need to for-loop otherwise some of them are empty
                cv2.fillPoly(lane_area_img, [lane_area], (255, 0, 0), **CV2_SUB_VALUES)

            lanes_area = lanes_area.reshape((-1, INTERPOLATION_POINTS, 2))
            for (
                name,
                mask,
            ) in lanes_mask.items():  # draw each type of lane with its own color
                cv2.polylines(
                    lane_line_img,
                    lanes_area[mask],
                    False,
                    (0, 255, 0),
                    **CV2_SUB_VALUES
                )

        # plot crosswalks
        crosswalks = []
        for idx in indices_in_bounds(
            center_in_world,
            self.mapAPI.bounds_info["crosswalks"]["bounds"],
            raster_radius,
        ):
            crosswalk = self.mapAPI.get_crosswalk_coords(
                self.mapAPI.bounds_info["crosswalks"]["ids"][idx]
            )
            xy_cross = cv2_subpixel(
                transform_points(crosswalk["xyz"][:, :2], raster_from_world)
            )
            crosswalks.append(xy_cross)

        cv2.fillPoly(ped_area_img, crosswalks, (0, 0, 255), **CV2_SUB_VALUES)

        map_img: np.ndarray = (lane_area_img + lane_line_img + ped_area_img).astype(
            np.float32
        ) / 255
        return map_img.transpose(2, 0, 1)
