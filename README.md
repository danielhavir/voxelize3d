# 3D Point Cloud Voxelization in NumPy

This is very similar to [spconv](https://github.com/traveller59/spconv) 's [voxelization implementation](https://github.com/traveller59/spconv/blob/master/spconv/utils/__init__.py)
but in plain NumPy without any C/C++ with on-par performance.

There are 2 versions:
* [Basic NumPy version](voxelize.py)
* [Numba-friendly JIT version](jit_voxelize.py)

## Example

```python
import numpy as np
from voxelize import voxelize
from jit_voxelize import voxelize_jit

points = ...  # (num_points, num_features) first 3 features must be x, y, z
voxels, coords, num_points_per_voxel = voxelize(points, voxel_size=np.array([0.2, 0.2, 0.4]), grid_range=np.array([-50, -50, -5, 50, 50, 3]), max_points_in_voxel=10, max_num_voxels=50000)

voxels, coords, num_points_per_voxel = voxelize_jit(points, voxel_size=np.array([0.2, 0.2, 0.4]), grid_range=np.array([-50, -50, -5, 50, 50, 3]), max_points_in_voxel=10, max_num_voxels=50000)
```

