from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trajdata.maps.vec_map import VectorMap

from typing import Optional

import numpy as np
from scipy.spatial import KDTree
from tqdm import tqdm

from trajdata.maps.vec_map_elements import MapElement, MapElementType, Polyline


class MapElementKDTree:
    """
    Constructs a KDTree of MapElements and exposes fast lookup functions.

    Inheriting classes need to implement the _extra_points function that defines for a MapElement
    the coordinates we want to store in the KDTree.
    """

    def __init__(self, vector_map: VectorMap) -> None:
        # Build kd-tree
        self.kdtree, self.polyline_inds = self._build_kdtree(vector_map)

    def _build_kdtree(self, vector_map: VectorMap):
        polylines = []
        polyline_inds = []

        map_elem: MapElement
        for map_elem in tqdm(
            vector_map.iter_elems(),
            desc=f"Building K-D Trees",
            leave=False,
            total=len(vector_map),
        ):
            points = self._extract_points(map_elem)
            if points is not None:
                polyline_inds.extend([len(polylines)] * points.shape[0])

                # Apply any map offsets to ensure we're in the same coordinate area as the
                # original world map.
                polylines.append(points)

        points = np.concatenate(polylines, axis=0)
        polyline_inds = np.array(polyline_inds)

        kdtree = KDTree(points)
        return kdtree, polyline_inds

    def _extract_points(self, map_element: MapElement) -> Optional[np.ndarray]:
        """Defines the coordinates we want to store in the KDTree for a MapElement.
        Args:
            map_element (MapElement): the MapElement to store in the KDTree.
        Returns:
            Optional[np.ndarray]: coordinates based on which we can search the KDTree, or None.
                If None, the MapElement will not be stored.
        """
        raise NotImplementedError()

    def closest_point(self, query_points: np.ndarray) -> np.ndarray:
        """Find the closest KDTree points to (a batch of) query points.

        Args:
            query_points: np.ndarray of shape (..., data_dim).

        Return:
            np.ndarray of shape (..., data_dim), the KDTree points closest to query_point.
        """
        _, data_inds = self.kdtree.query(query_points, k=1)
        pts = self.kdtree.data[data_inds]
        return pts

    def closest_polyline_ind(self, query_points: np.ndarray) -> np.ndarray:
        """Find the index of the closest polyline(s) in self.polylines."""
        _, data_ind = self.kdtree.query(query_points, k=1)
        return self.polyline_inds[data_ind]

    def polyline_inds_in_range(self, point: np.ndarray, range: float) -> np.ndarray:
        """Find the index of polylines in self.polylines within 'range' distance to 'point'."""
        data_inds = self.kdtree.query_ball_point(point, range)
        return np.unique(self.polyline_inds[data_inds], axis=0)


class LaneCenterKDTree(MapElementKDTree):
    """KDTree for lane center polylines."""

    def __init__(
        self, vector_map: VectorMap, max_segment_len: Optional[float] = None
    ) -> None:
        """
        Args:
            vec_map: the VectorizedMap object to build the KDTree for
            max_segment_len (float, optional): if specified, we will insert extra points into the KDTree
                such that all polyline segments are shorter then max_segment_len.
        """
        self.max_segment_len = max_segment_len
        super().__init__(vector_map)

    def _extract_points(self, map_element: MapElement) -> Optional[np.ndarray]:
        if map_element.elem_type == MapElementType.ROAD_LANE:
            pts: Polyline = map_element.center
            if self.max_segment_len is not None:
                pts = pts.interpolate(max_dist=self.max_segment_len)

            return pts.points
        else:
            return None
