from collections import defaultdict

from torch.utils.data import DataLoader
from tqdm import tqdm

from trajdata import AgentBatch, AgentType, UnifiedDataset
from trajdata.visualization.interactive_animation import (
    InteractiveAnimation,
    animate_agent_batch_interactive,
)
from trajdata.visualization.interactive_vis import plot_agent_batch_interactive
from trajdata.visualization.vis import plot_agent_batch


def main():
    dataset = UnifiedDataset(
        desired_data=["nusc_mini"],
        centric="agent",
        desired_dt=0.1,
        # history_sec=(3.2, 3.2),
        # future_sec=(4.8, 4.8),
        only_predict=[AgentType.VEHICLE],
        state_format="x,y,z,xd,yd,h",
        obs_format="x,y,z,xd,yd,s,c",
        # agent_interaction_distances=defaultdict(lambda: 30.0),
        incl_robot_future=False,
        incl_raster_map=True,
        raster_map_params={
            "px_per_m": 2,
            "map_size_px": 224,
            "offset_frac_xy": (-0.5, 0.0),
        },
        num_workers=4,
        verbose=True,
        data_dirs={  # Remember to change this to match your filesystem!
            "nusc_mini": "~/datasets/nuScenes",
            "lyft_sample": "~/datasets/lyft/scenes/sample.zarr",
            "nuplan_mini": "~/datasets/nuplan/dataset/nuplan-v1.1",
        },
    )

    print(f"# Data Samples: {len(dataset):,}")

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=dataset.get_collate_fn(),
        num_workers=0,
    )

    batch: AgentBatch
    for batch in tqdm(dataloader):
        plot_agent_batch_interactive(batch, batch_idx=0, cache_path=dataset.cache_path)
        plot_agent_batch(batch, batch_idx=0)

        animation = InteractiveAnimation(
            animate_agent_batch_interactive,
            batch=batch,
            batch_idx=0,
            cache_path=dataset.cache_path,
        )
        animation.show()
        # break


if __name__ == "__main__":
    main()
