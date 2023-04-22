# Supported Datasets and Required Formats

## nuScenes
Nothing special needs to be done for the nuScenes dataset, simply install it as per [the instructions in the devkit README](https://github.com/nutonomy/nuscenes-devkit#nuscenes-setup).

It should look like this after downloading:
```
/path/to/nuScenes/
            ├── maps/
            ├── samples/
            ├── sweeps/
            ├── v1.0-mini/
            ├── v1.0-test/
            └── v1.0-trainval/
```

**Note**: At a minimum, only the annotations need to be downloaded (not the raw radar/camera/lidar/etc data).

## nuPlan
Nothing special needs to be done for the nuPlan dataset, simply download v1.1 as per [the instructions in the devkit documentation](https://nuplan-devkit.readthedocs.io/en/latest/dataset_setup.html).

It should look like this after downloading:
```
/path/to/nuPlan/
            └── dataset
                ├── maps
                │   ├── nuplan-maps-v1.0.json
                │   ├── sg-one-north
                │   │   └── 9.17.1964
                │   │       └── map.gpkg
                │   ├── us-ma-boston
                │   │   └── 9.12.1817
                │   │       └── map.gpkg
                │   ├── us-nv-las-vegas-strip
                │   │   └── 9.15.1915
                │   │       ├── drivable_area.npy.npz
                │   │       ├── Intensity.npy.npz
                │   │       └── map.gpkg
                │   └── us-pa-pittsburgh-hazelwood
                │       └── 9.17.1937
                │           └── map.gpkg
                └── nuplan-v1.1
                    ├── mini
                    │   ├── 2021.05.12.22.00.38_veh-35_01008_01518.db
                    │   ├── 2021.06.09.17.23.18_veh-38_00773_01140.db
                    │   ├── ...
                    │   └── 2021.10.11.08.31.07_veh-50_01750_01948.db
                    └── trainval
                        ├── 2021.05.12.22.00.38_veh-35_01008_01518.db
                        ├── 2021.06.09.17.23.18_veh-38_00773_01140.db
                        ├── ...
                        └── 2021.10.11.08.31.07_veh-50_01750_01948.db
```

**Note**: Not all dataset splits need to be downloaded. For example, you can download only the nuPlan Mini Split in case you only need a small sample dataset.

## Lyft Level 5
Nothing special needs to be done for the Lyft Level 5 dataset, simply install it as per [the instructions on the dataset website](https://woven-planet.github.io/l5kit/dataset.html).

It should look like this after downloading:
```
/path/to/lyft/
            ├── LICENSE
            ├── aerial_map
            ├── feedback.txt
            ├── meta.json
            ├── scenes/
            │   ├── sample.zarr
            |   ├── train.zarr
            |   └── ...
            └── semantic_map/
                └── semantic_map.pb
```

**Note**: Not all the dataset parts need to be downloaded, only the necessary `.zarr` files need to be downloaded (e.g., `sample.zarr` for the small sample dataset).

## ETH/UCY Pedestrians
The raw data can be found in many places online, ranging from [research projects' data download scripts](https://github.com/agrimgupta92/sgan/blob/master/scripts/download_data.sh) to [copies of the original data itself](https://github.com/StanfordASL/Trajectron-plus-plus/tree/master/experiments/pedestrians/raw/raw/all_data) on GitHub. In this data loader, we assume the data was sourced from the latter.

It should look like this after downloading:
```
/path/to/eth_ucy/
            ├── biwi_eth.txt
            ├── biwi_hotel.txt
            ├── crowds_zara01.txt
            ├── crowds_zara02.txt
            ├── crowds_zara03.txt
            ├── students001.txt
            ├── students003.txt
            └── uni_examples.txt
```
