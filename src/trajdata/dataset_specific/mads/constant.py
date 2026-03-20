# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Constants and enums for the MADS dataset loader."""

import os
from enum import Enum
from typing import Final, FrozenSet, Optional

# Core dataset timing/config constants.
MADS_DT: Final[float] = 0.1
EGO_LENGTH: Final[float] = 5.2993629
EGO_WIDTH: Final[float] = 2.11311007
EGO_HEIGHT: Final[float] = 1.34435794

# by default, we support internal production v2 dataset, with clipgt-2.0.0 and above.
DATA_SRC: Final[str] = "v2"

# Allowed MADS data source tags.
SUPPORTED_DATA_SRCS: Final[FrozenSet[str]] = frozenset({"v2", "pai"})

# Environment variable name for runtime override.
DATA_SRC_ENV_VAR: Final[str] = "MADS_DATA_SRC"


def resolve_data_src(cli_override: Optional[str] = None) -> str:
    """Resolve dataset source with priority: CLI override > env var > default."""
    candidate = (cli_override or os.getenv(DATA_SRC_ENV_VAR) or DATA_SRC).strip()
    if candidate not in SUPPORTED_DATA_SRCS:
        supported = ", ".join(sorted(SUPPORTED_DATA_SRCS))
        raise ValueError(
            f"Unsupported data source '{candidate}'. "
            f"Supported values: {supported}."
        )
    return candidate

# Minimum frames for an agent to be considered.
MIN_FRAMES: Final[int] = 10

USE_CUBIC_INTERPOLATION: Final[bool] = False


class ObstacleClassV1(Enum):
    """Obstacle classes for MADS based on NDAS `obstacle_types.proto`."""

    # buf:lint:ignore ENUM_ZERO_VALUE_SUFFIX
    OBSTACLE_CLASS_INVALID = 0
    # Vehicles
    OBSTACLE_CLASS_VEHICLE_UNKNOWN = 1280
    OBSTACLE_CLASS_VEHICLE_CAR = 1281
    OBSTACLE_CLASS_VEHICLE_TRUCK = 1282
    OBSTACLE_CLASS_VEHICLE_BUS = 1283
    OBSTACLE_CLASS_VEHICLE_EMERGENCY = 1284
    OBSTACLE_CLASS_VEHICLE_CONSTRUCTION = 1285
    OBSTACLE_CLASS_VEHICLE_POLICE = 1286
    OBSTACLE_CLASS_VEHICLE_SCHOOL_BUS = 1287
    # Two wheeled vehicles
    OBSTACLE_CLASS_BIKE_UNKNOWN = 2304
    OBSTACLE_CLASS_BIKE = 2305
    OBSTACLE_CLASS_BIKE_MOTOR = 2306
    # Two wheeled vehicles with rider
    OBSTACLE_CLASS_BIKE_UNKNOWN_WITH_RIDER = 2307
    OBSTACLE_CLASS_BIKE_WITH_RIDER = 2308
    OBSTACLE_CLASS_BIKE_MOTOR_WITH_RIDER = 2309
    OBSTACLE_CLASS_TRICYCLE = 2310
    # Pedestrians
    OBSTACLE_CLASS_PEDESTRIAN_UNKNOWN = 4352
    OBSTACLE_CLASS_PEDESTRIAN_ADULT = 4353
    OBSTACLE_CLASS_PEDESTRIAN_CHILD = 4354
    OBSTACLE_CLASS_PEDESTRIAN_CONSTRUCTION_WORKER = 4355
    OBSTACLE_CLASS_PEDESTRIAN_OFFICIAL = 4356
    # Animals
    OBSTACLE_CLASS_ANIMAL_UNKNOWN = 8448
    OBSTACLE_CLASS_ANIMAL_SMALL = 8449
    OBSTACLE_CLASS_ANIMAL_MEDIUM = 8450
    OBSTACLE_CLASS_ANIMAL_LARGE = 8451
    # Objects
    OBSTACLE_CLASS_OBJECT_UNKNOWN = 16384
    OBSTACLE_CLASS_OBJECT_HAZARD = 16385
    OBSTACLE_CLASS_OBJECT_OFFICIAL = 16386
    OBSTACLE_CLASS_OBJECT_CONE = 16387
    OBSTACLE_CLASS_OBJECT_CURB = 16388
    OBSTACLE_CLASS_OBJECT_BARRIER = 16389
    OBSTACLE_CLASS_OBJECT_PHYSICAL_LANE_LINE = 16390
    OBSTACLE_CLASS_OBJECT_HARD_OVERHANG = 16391
    OBSTACLE_CLASS_OBJECT_SOFT_OVERHANG = 16392
    OBSTACLE_CLASS_OBJECT_SPEED_BUMP = 16393
    OBSTACLE_CLASS_OBJECT_POT_HOLE = 16394
    OBSTACLE_CLASS_OBJECT_NEGATIVE = 16395
    OBSTACLE_CLASS_OBJECT_FORBIDDEN = 16396
    # Other
    OBSTACLE_CLASS_UNDEFINED_STATIC = 32768
    OBSTACLE_CLASS_UNDEFINED_ANIMATE = 33024
    # Virtual obstacles
    OBSTACLE_CLASS_FORBIDDEN_LANE_LINE = 513
    OBSTACLE_CLASS_CROSSABLE_LANE_LINE = 514
    OBSTACLE_CLASS_GORE_AREA = 515
    OBSTACLE_CLASS_OCCLUSION = 516

    @classmethod
    def get_enum_name(cls, value: int) -> Optional[str]:
        """Return enum member name for a numeric value, or `None` if not found."""
        for name, member in cls.__members__.items():
            if member.value == value:
                return name
        return None
