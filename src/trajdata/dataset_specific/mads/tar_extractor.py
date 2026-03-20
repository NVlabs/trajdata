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
#

"""Tar extraction utilities for component data files.

Provides TarExtractor class for extracting tar archives to RAM filesystem
for fast access during data processing operations.
"""

import pathlib
import shutil
import tarfile
import uuid
from typing import Optional


class TarExtractor:
    """Extracts tar archives to /dev/shm for fast access."""

    def __init__(self):
        """Initialize extractor with empty state, ready to extract a tar file."""
        self._filename: Optional[pathlib.Path] = None
        self._extracted_tar_file: Optional[pathlib.Path] = None
        self._temp_dir: Optional[pathlib.Path] = None

    def get_clip_dir(self, tar_filepath: pathlib.Path) -> pathlib.Path:
        """Get directory with extracted components.

        Args:
            base_path: Directory containing the tar file.

        Returns:
            Path to extracted components directory.
        """
        tar_filepath = pathlib.Path(tar_filepath)

        # Return cached directory if already extracted
        if self._extracted_tar_file == tar_filepath and self._temp_dir:
            return self._temp_dir

        self._extract(tar_filepath)
        if self._temp_dir is None:
            raise RuntimeError(f"Failed to extract tar file: {tar_filepath}")
        return self._temp_dir

    def _extract(self, tar_file: pathlib.Path):
        """Extract tar to temporary directory in /dev/shm.

        Args:
            tar_file: Path to tar file to extract.
        """
        self.cleanup()

        # Create temp directory
        clip_id = tar_file.parent.name
        untar_id = uuid.uuid4().hex[:10]
        self._temp_dir = (
            pathlib.Path("/dev/shm") / f"untar_{untar_id}_of_clip_{clip_id}"
        )
        self._temp_dir.mkdir(exist_ok=True)

        try:
            with tarfile.open(tar_file, "r") as tar:
                tar.extractall(self._temp_dir)
            self._extracted_tar_file = tar_file
        except Exception:
            self.cleanup()
            raise

    def cleanup(self):
        """Remove temporary directory and reset state."""
        if self._temp_dir:
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None
        self._extracted_tar_file = None

    def __del__(self):
        """Clean up temporary files when object is destroyed."""
        self.cleanup()
