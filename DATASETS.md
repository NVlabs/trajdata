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
