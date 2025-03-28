import numpy as np
from numba import njit


@njit
def voxelize_jit(
        points: np.ndarray,
        voxel_size: np.ndarray,
        grid_range: np.ndarray,
        max_points_in_voxel: int = 60,
        max_num_voxels: int = 20000
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Numba-friendly version of voxelize
    :param points: (num_points, num_features), first 3 elements must be <x>, <y>, <z>
    :param voxel_size: (3,) - <width>, <length>, <height>
    :param grid_range: (6,) - <min_x>, <min_y>, <min_z>, <max_x>, <max_y>, <max_z>
    :param max_points_in_voxel:
    :param max_num_voxels:
    :return: tuple (
        voxels (num_voxels, max_points_in_voxels, num_features),
        coordinates (num_voxels, 3),
        num_points_per_voxel (num_voxels,)
    )
    """
    points_copy = points.copy()
    grid_size = np.floor((grid_range[3:] - grid_range[:3]) / voxel_size).astype(np.int32)

    coor_to_voxelidx = np.full((grid_size[2], grid_size[1], grid_size[0]), -1, dtype=np.int32)
    voxels = np.zeros((max_num_voxels, max_points_in_voxel, points_copy.shape[-1]), dtype=points_copy.dtype)
    coors = np.zeros((max_num_voxels, 3), dtype=np.int32)
    num_points_per_voxel = np.zeros(shape=(max_num_voxels,), dtype=np.int32)

    coor = np.floor((points_copy[:, :3] - grid_range[:3]) / voxel_size).astype(np.int32)
    mask = np.logical_and(np.logical_and((coor[:, 0] >= 0) & (coor[:, 0] < grid_size[0]),
                                         (coor[:, 1] >= 0) & (coor[:, 1] < grid_size[1])),
                          (coor[:, 2] >= 0) & (coor[:, 2] < grid_size[2]))
    coor = coor[mask, ::-1]
    points_copy = points_copy[mask]
    assert points_copy.shape[0] == coor.shape[0]

    voxel_num = 0
    for i, c in enumerate(coor):
        voxel_id = coor_to_voxelidx[c[0], c[1], c[2]]
        if voxel_id == -1:
            voxel_id = voxel_num
            voxel_num += 1
            if voxel_num > max_num_voxels:
                break
            coor_to_voxelidx[c[0], c[1], c[2]] = voxel_id
            coors[voxel_id] = c
        n_pts = num_points_per_voxel[voxel_id]
        if n_pts < max_points_in_voxel:
            voxels[voxel_id, n_pts] = points_copy[i]
            num_points_per_voxel[voxel_id] += 1

    return voxels[:voxel_num], coors[:voxel_num], num_points_per_voxel[:voxel_num]
