# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import glob
import os

import numpy as np
import torch


def load_amp_data(data_path: str | list[str], device: str = "cpu") -> dict:
    paths = [data_path] if isinstance(data_path, str) else data_path
    offline_data = []

    if not isinstance(paths, list):
        raise ValueError("data_path must be a str or a list of str")

    # Expand glob patterns and collect all files
    all_files = set()
    for path in paths:
        matching_paths = glob.glob(path)
        for matched_path in matching_paths:
            if os.path.isfile(matched_path):
                all_files.add(os.path.abspath(matched_path))
    if not all_files:
        raise FileNotFoundError(f"No files found matching patterns: {paths}")

    # Sort files for consistent loading order
    all_files = sorted(all_files)

    # Load all files
    for file_path in all_files:
        data = np.load(file_path).astype(np.float32)
        offline_data.append(torch.from_numpy(data).to(device))

    # Concatenate all data
    if len(offline_data) == 1:
        motion_data = offline_data[0]
    else:
        motion_data = torch.cat(offline_data, dim=0)

    # Generate motion_step indices
    motion_step = torch.arange(motion_data.shape[0], device=device, dtype=torch.long)

    # If multiple files were loaded, adjust motion_step to reset for each file
    if len(offline_data) > 1:
        step_list = []
        for data_chunk in offline_data:
            step_list.append(torch.arange(data_chunk.shape[0], device=device, dtype=torch.long))
        motion_step = torch.cat(step_list, dim=0)

    # Return dictionary as required by AMP algorithm
    return {
        "motion_data": motion_data,
        "motion_step": motion_step,
    }
