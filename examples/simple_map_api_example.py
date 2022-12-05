import time
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from trajdata import MapAPI, VectorMap


def main():
    cache_path = Path("~/.unified_data_cache").expanduser()
    map_api = MapAPI(cache_path)

    ### Loading random scene and initializing VectorMap.
    env_name: str = np.random.choice(["nusc_mini", "lyft_sample", "nuplan_mini"])
    random_location_dict: Dict[str, str] = {
        "nuplan_mini": np.random.choice(
            ["boston", "singapore", "pittsburgh", "las_vegas"]
        ),
        "nusc_mini": np.random.choice(["boston-seaport", "singapore-onenorth"]),
        "lyft_sample": "palo_alto",
    }

    start = time.perf_counter()
    vec_map: VectorMap = map_api.get_map(f"{env_name}:{random_location_dict[env_name]}")
    end = time.perf_counter()
    print(f"Map loading took {(end - start)*1000:.2f} ms")

    start = time.perf_counter()
    vec_map: VectorMap = map_api.get_map(f"{env_name}:{random_location_dict[env_name]}")
    end = time.perf_counter()
    print(f"Repeated (cached in memory) map loading took {(end - start)*1000:.2f} ms")

    print(f"Randomly chose {vec_map.env_name}, {vec_map.map_name} map.")

    ### Lane Graph Visualization (with rasterized map in background)
    fig, ax = plt.subplots()

    print(f"Rasterizing Map...")
    start = time.perf_counter()
    map_img, raster_from_world = vec_map.rasterize(
        resolution=2,
        return_tf_mat=True,
        incl_centerlines=False,
        area_color=(255, 255, 255),
        edge_color=(0, 0, 0),
        scene_ts=100,
    )
    end = time.perf_counter()
    print(f"Map rasterization took {(end - start)*1000:.2f} ms")

    ax.imshow(map_img, alpha=0.5, origin="lower")

    lane_idx = np.random.randint(0, len(vec_map.lanes))
    print(f"Visualizing random lane index {lane_idx}...")
    start = time.perf_counter()
    vec_map.visualize_lane_graph(
        origin_lane=lane_idx,
        num_hops=10,
        raster_from_world=raster_from_world,
        ax=ax,
    )
    end = time.perf_counter()
    print(f"Lane visualization took {(end - start)*1000:.2f} ms")

    ax.axis("equal")
    ax.grid(None)

    plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()
