# 3D Point Cloud Voxelization in NumPy

This is very similar to [spconv](https://github.com/traveller59/spconv) 's [voxelization implementation](https://github.com/traveller59/spconv/blob/master/spconv/utils/__init__.py)
but in plain NumPy without any C/C++ with on-par performance.

There are 2 versions:
* [Basic NumPy version](voxelize.py)
* [Numba-friendly JIT version](jit_voxelize.py)
