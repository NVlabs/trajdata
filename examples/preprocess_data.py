import os

from trajdata import UnifiedDataset


def main():
    dataset = UnifiedDataset(
        desired_data=["waymo_val"],
        rebuild_cache=True,
        rebuild_maps=True,
        num_workers=os.cpu_count(),
        verbose=True,
        data_dirs={  # Remember to change this to match your filesystem!
            "waymo_val": "~/datasets/waymo",
        },
    )
    print(f"Total Data Samples: {len(dataset):,}")


if __name__ == "__main__":
    main()
