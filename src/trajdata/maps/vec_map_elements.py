from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional, Set

import numpy as np

from trajdata.utils import map_utils


class MapElementType(IntEnum):
    ROAD_LANE = 1
    ROAD_AREA = 2
    PED_CROSSWALK = 3
    PED_WALKWAY = 4


@dataclass
class Polyline:
    points: np.ndarray

    def __post_init__(self) -> None:
        if self.points.shape[-1] < 2:
            raise ValueError(
                f"Polylines are expected to have 2 (xy), 3 (xyz), or 4 (xyzh) dimensions, but received {self.points.shape[-1]}."
            )

        if self.points.shape[-1] == 2:
            # If only xy are passed in, then append zero to the end for z.
            self.points = np.append(
                self.points, np.zeros_like(self.points[:, [0]]), axis=-1
            )

    @property
    def midpoint(self) -> np.ndarray:
        num_pts: int = self.points.shape[0]
        return self.points[num_pts // 2]

    @property
    def has_heading(self) -> bool:
        return self.points.shape[-1] == 4

    @property
    def xy(self) -> np.ndarray:
        return self.points[..., :2]

    @property
    def xyz(self) -> np.ndarray:
        return self.points[..., :3]

    @property
    def xyzh(self) -> np.ndarray:
        if self.has_heading:
            return self.points[..., :4]
        else:
            raise ValueError(
                f"This Polyline only has {self.points.shape[-1]} coordinates, expected 4."
            )

    @property
    def h(self) -> np.ndarray:
        return self.points[..., 3]

    def interpolate(
        self, num_pts: Optional[int] = None, max_dist: Optional[float] = None
    ) -> "Polyline":
        return Polyline(
            map_utils.interpolate(self.points, num_pts=num_pts, max_dist=max_dist)
        )

    def project_onto(self, xyz_or_xyzh: np.ndarray) -> np.ndarray:
        """Project the given points onto this Polyline.

        Args:
            xyzh (np.ndarray): Points to project, of shape (M, D)

        Returns:
            np.ndarray: The projected points, of shape (M, D)

        Note:
            D = 4 if this Polyline has headings, otherwise D = 3
        """
        # xyzh is now (M, 1, 3), we do not use heading for projection.
        xyz = xyz_or_xyzh[:, np.newaxis, :3]

        # p0, p1 are (1, N, 3)
        p0: np.ndarray = self.points[np.newaxis, :-1, :3]
        p1: np.ndarray = self.points[np.newaxis, 1:, :3]

        # 1. Compute projections of each point to each line segment in a
        #    batched manner.
        line_seg_diffs: np.ndarray = p1 - p0
        point_seg_diffs: np.ndarray = xyz - p0

        dot_products: np.ndarray = (point_seg_diffs * line_seg_diffs).sum(
            axis=-1, keepdims=True
        )
        norms: np.ndarray = np.linalg.norm(line_seg_diffs, axis=-1, keepdims=True) ** 2

        # Clip ensures that the projected point stays within the line segment boundaries.
        projs: np.ndarray = (
            p0 + np.clip(dot_products / norms, a_min=0, a_max=1) * line_seg_diffs
        )

        # 2. Find the nearest projections to the original points.
        closest_proj_idxs: int = np.linalg.norm(xyz - projs, axis=-1).argmin(axis=-1)

        if self.has_heading:
            # Adding in the heading of the corresponding p0 point (which makes
            # sense as p0 to p1 is a line => same heading along it).
            return np.concatenate(
                [
                    projs[range(xyz.shape[0]), closest_proj_idxs],
                    np.expand_dims(self.points[closest_proj_idxs, -1], axis=-1),
                ],
                axis=-1,
            )
        else:
            return projs[range(xyz.shape[0]), closest_proj_idxs]


@dataclass
class MapElement:
    id: str


@dataclass
class RoadLane(MapElement):
    center: Polyline
    left_edge: Optional[Polyline] = None
    right_edge: Optional[Polyline] = None
    adj_lanes_left: Set[str] = field(default_factory=lambda: set())
    adj_lanes_right: Set[str] = field(default_factory=lambda: set())
    next_lanes: Set[str] = field(default_factory=lambda: set())
    prev_lanes: Set[str] = field(default_factory=lambda: set())
    elem_type: MapElementType = MapElementType.ROAD_LANE

    def __post_init__(self) -> None:
        if not self.center.has_heading:
            self.center = Polyline(
                np.append(
                    self.center.xyz,
                    map_utils.get_polyline_headings(self.center.xyz),
                    axis=-1,
                )
            )

    def __hash__(self) -> int:
        return hash(self.id)

    @property
    def reachable_lanes(self) -> Set[str]:
        return self.adj_lanes_left | self.adj_lanes_right | self.next_lanes


@dataclass
class RoadArea(MapElement):
    exterior_polygon: Polyline
    interior_holes: List[Polyline] = field(default_factory=lambda: list())
    elem_type: MapElementType = MapElementType.ROAD_AREA


@dataclass
class PedCrosswalk(MapElement):
    polygon: Polyline
    elem_type: MapElementType = MapElementType.PED_CROSSWALK


@dataclass
class PedWalkway(MapElement):
    polygon: Polyline
    elem_type: MapElementType = MapElementType.PED_WALKWAY
