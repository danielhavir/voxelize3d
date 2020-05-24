from typing import Tuple

import numpy as np


def voxelize(
        points: np.ndarray,
        voxel_size: np.ndarray,
        grid_range: np.ndarray,
        max_points_in_voxel: int,
        max_num_voxels: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Converts 3D point cloud to a sparse voxel grid
    :param points: (num_points, num_features), first 3 elements must be <x>, <y>, <z>
    :param voxel_size: (3,) - <width>, <length>, <height>
    :param grid_range: (6,) - <min_x>, <min_y>, <min_z>, <max_x>, <max_y>, <max_z>
    :param max_points_in_voxel:
    :param max_num_voxels:
    :param include_relative_position: boolean flag, if True, the output num_features will include relative
    position of the point within the voxel
    :return: tuple (
        voxels (num_voxels, max_points_in_voxels, num_features),
        coordinates (num_voxels, 3),
        num_points_per_voxel (num_voxels,)
    )
    """
    points_copy = points.copy()
    grid_size = np.floor((grid_range[3:] - grid_range[:3]) / voxel_size).astype(np.int32)

    coor_to_voxelidx = np.full((grid_size[2], grid_size[1], grid_size[0]), -1, dtype=np.int32)
    voxels = np.zeros((max_num_voxels, max_points_in_voxel, points.shape[-1]), dtype=points_copy.dtype)
    coordinates = np.zeros((max_num_voxels, 3), dtype=np.int32)
    num_points_per_voxel = np.zeros(max_num_voxels, dtype=np.int32)

    points_coords = np.floor((points_copy[:, :3] - grid_range[:3]) / voxel_size).astype(np.int32)
    mask = ((points_coords >= 0) & (points_coords < grid_size)).all(1)
    points_coords = points_coords[mask, ::-1]
    points_copy = points_copy[mask]
    assert points_copy.shape[0] == points_coords.shape[0]

    voxel_num = 0
    for i, coord in enumerate(points_coords):
        voxel_idx = coor_to_voxelidx[tuple(coord)]
        if voxel_idx == -1:
            voxel_idx = voxel_num
            if voxel_num > max_num_voxels:
                continue
            voxel_num += 1
            coor_to_voxelidx[tuple(coord)] = voxel_idx
            coordinates[voxel_idx] = coord
        point_idx = num_points_per_voxel[voxel_idx]
        if point_idx < max_points_in_voxel:
            voxels[voxel_idx, point_idx] = points_copy[i]
            num_points_per_voxel[voxel_idx] += 1

    return voxels[:voxel_num], coordinates[:voxel_num], num_points_per_voxel[:voxel_num]
