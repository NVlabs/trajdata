"""Lane connectivity and junction handling for XODR parsing.

This module handles road-to-road connectivity, lane adjacency,
and junction connections.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Dict, List

import numpy as np

from .datatypes import LaneGeom

# Constants
# Maximum distance (meters) to connect junction lanes
JUNCTION_LANE_CONNECTION_THRESHOLD = 10.0


def build_lane_adjacency(road_to_lanes: Dict[str, List[LaneGeom]]) -> None:
    """Build adjacency relationships between lanes within each road.

    Args:
        road_to_lanes: Dictionary mapping road IDs to their lane geometries.
                      Modified in place to add adjacency relationships.
    """
    for lanes in road_to_lanes.values():
        left_lanes = sorted(
            [lane for lane in lanes if lane.is_left], key=lambda lane: lane.lane_id_xml
        )
        right_lanes = sorted(
            [lane for lane in lanes if not lane.is_left],
            key=lambda lane: lane.lane_id_xml,
            reverse=True,
        )
        for side in (left_lanes, right_lanes):
            for i, curr in enumerate(side):
                if i > 0:
                    curr.adj_right.add(side[i - 1].unique_id)
                if i < len(side) - 1:
                    curr.adj_left.add(side[i + 1].unique_id)

        # Populate legal lane change permissions from XODR neighbor references
        # IMPORTANT: Only allow lane changes that are BOTH legal AND physically possible
        for lane in lanes:
            # For most lanes, only forward neighbors are relevant
            # Intersect with geometric adjacency to ensure physical reachability
            lane.can_change_left.update(lane.left_neighbor_forward & lane.adj_left)
            lane.can_change_right.update(lane.right_neighbor_forward & lane.adj_right)

            # For lanes with direction="both", backward neighbors are also important
            # (vehicles can legally travel in either direction)
            if lane.direction == "both":
                # When traveling backward, left becomes right and vice versa
                lane.can_change_right.update(
                    lane.left_neighbor_backward & lane.adj_right
                )
                lane.can_change_left.update(
                    lane.right_neighbor_backward & lane.adj_left
                )


def extract_road_connections(root: ET.Element) -> Dict[str, Dict]:
    """Extract road-to-road connectivity from <link> elements.

    Args:
        root: Root XML element of the XODR document

    Returns:
        Dictionary mapping road IDs to their connectivity info
    """
    road_connections: Dict[str, Dict] = {}

    for road in root.findall("road"):
        road_id = road.attrib["id"]
        link_elem = road.find("link")
        if link_elem is not None:
            connections = {}
            predecessor = link_elem.find("predecessor")
            successor = link_elem.find("successor")

            if (
                predecessor is not None
                and predecessor.attrib.get("elementType") == "road"
            ):
                connections["predecessor"] = {
                    "road_id": predecessor.attrib["elementId"],
                    "contact": predecessor.attrib.get("contactPoint", "start"),
                }

            if successor is not None and successor.attrib.get("elementType") == "road":
                connections["successor"] = {
                    "road_id": successor.attrib["elementId"],
                    "contact": successor.attrib.get("contactPoint", "start"),
                }

            if connections:
                road_connections[road_id] = connections

    return road_connections


def connect_lanes_between_roads(
    root: ET.Element,
    road_connections: Dict[str, Dict],
    road_to_lanes: Dict[str, List[LaneGeom]],
) -> None:
    """Connect lanes between roads using connectivity information.

    Handles both regular road connections (exact lane ID match) and
    junction connections (spatial proximity).

    Args:
        root: Root XML element of the XODR document
        road_connections: Road connectivity information
        road_to_lanes: Dictionary mapping road IDs to their lanes
                      Modified in place to add lane connections.
    """
    for road_id, connections in road_connections.items():
        current_road_lanes = road_to_lanes.get(road_id, [])

        for direction, conn_info in connections.items():
            connected_road_id = conn_info["road_id"]
            connected_road_lanes = road_to_lanes.get(connected_road_id, [])

            # Determine if this is a junction connection
            curr_road = next(
                (r for r in root.findall("road") if r.attrib["id"] == road_id), None
            )
            conn_road = next(
                (
                    r
                    for r in root.findall("road")
                    if r.attrib["id"] == connected_road_id
                ),
                None,
            )

            curr_is_junction = (
                curr_road is not None and curr_road.attrib.get("junction", "-1") != "-1"
            )
            conn_is_junction = (
                conn_road is not None and conn_road.attrib.get("junction", "-1") != "-1"
            )

            if curr_is_junction or conn_is_junction:
                # Junction connection: use spatial proximity
                _connect_junction_lanes(
                    current_road_lanes,
                    connected_road_lanes,
                    direction,
                )
            else:
                # Regular road connection: use exact lane ID match
                _connect_regular_lanes(
                    current_road_lanes,
                    connected_road_lanes,
                    direction,
                )


def _connect_junction_lanes(
    current_road_lanes: List[LaneGeom],
    connected_road_lanes: List[LaneGeom],
    direction: str,
) -> None:
    """Connect lanes in junction areas using spatial proximity.

    Args:
        current_road_lanes: Lanes from current road
        connected_road_lanes: Lanes from connected road
        direction: "successor" or "predecessor"
    """
    for curr_lane in current_road_lanes:
        if not curr_lane.is_driving:
            continue  # Only connect driveable lanes

        # Find closest lane in connected road (by end/start point proximity)
        curr_endpoint = (
            curr_lane.center[-1] if direction == "successor" else curr_lane.center[0]
        )

        best_match = None
        min_distance = float("inf")

        for conn_lane in connected_road_lanes:
            if not conn_lane.is_driving:
                continue

            # Check distance to appropriate endpoint
            conn_endpoint = (
                conn_lane.center[0]
                if direction == "successor"
                else conn_lane.center[-1]
            )
            distance = np.linalg.norm(curr_endpoint[:2] - conn_endpoint[:2])

            if (
                distance < min_distance
                and distance < JUNCTION_LANE_CONNECTION_THRESHOLD
            ):
                min_distance = distance
                best_match = conn_lane

        if best_match is not None:
            if direction == "successor":
                curr_lane.next_lanes.add(best_match.unique_id)
                best_match.prev_lanes.add(curr_lane.unique_id)
            elif direction == "predecessor":
                curr_lane.prev_lanes.add(best_match.unique_id)
                best_match.next_lanes.add(curr_lane.unique_id)


def _connect_regular_lanes(
    current_road_lanes: List[LaneGeom],
    connected_road_lanes: List[LaneGeom],
    direction: str,
) -> None:
    """Connect lanes between regular roads using exact lane ID match.

    Args:
        current_road_lanes: Lanes from current road
        connected_road_lanes: Lanes from connected road
        direction: "successor" or "predecessor"
    """
    for curr_lane in current_road_lanes:
        for conn_lane in connected_road_lanes:
            if curr_lane.lane_id_xml == conn_lane.lane_id_xml:  # Same lane number
                if direction == "successor":
                    curr_lane.next_lanes.add(conn_lane.unique_id)
                    conn_lane.prev_lanes.add(curr_lane.unique_id)
                elif direction == "predecessor":
                    curr_lane.prev_lanes.add(conn_lane.unique_id)
                    conn_lane.next_lanes.add(curr_lane.unique_id)


def process_junction_connections(
    root: ET.Element,
    lane_geoms: Dict[str, LaneGeom],
) -> None:
    """Process explicit junction laneLink connectivity.

    Args:
        root: Root XML element of the XODR document
        lane_geoms: Dictionary of all lane geometries
                   Modified in place to add junction connections.
    """
    for junction in root.findall("junction"):
        for conn in junction.findall("connection"):
            conn_road_id = conn.attrib["connectingRoad"]
            incoming_road_id = conn.attrib["incomingRoad"]
            contact_pt = conn.attrib.get(
                "contactPoint", "end"
            )  # 'start' or 'end' relative to incoming road

            for ll in conn.findall("laneLink"):
                from_lane = int(ll.attrib["from"])  # lane id in connectingRoad
                to_lane = int(ll.attrib["to"])  # lane id in incomingRoad

                unique_from = f"{conn_road_id}_{from_lane}"
                unique_to = f"{incoming_road_id}_{to_lane}"

                if unique_from not in lane_geoms or unique_to not in lane_geoms:
                    continue  # malformed reference, skip

                # Establish prev/next according to OpenDRIVE spec:
                # incomingRoad -> connectingRoad -> (other outgoing road)
                if contact_pt == "start":
                    # connectingRoad starts at incoming road, so incoming -> connecting
                    lane_geoms[unique_from].prev_lanes.add(unique_to)
                    lane_geoms[unique_to].next_lanes.add(unique_from)
                else:  # 'end'
                    # connectingRoad ends at incoming road, so connecting -> incoming
                    lane_geoms[unique_from].next_lanes.add(unique_to)
                    lane_geoms[unique_to].prev_lanes.add(unique_from)
