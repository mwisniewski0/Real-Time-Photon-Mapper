# CUDA common

This directory contains CUDA specific code used throughout the project.

* **[gpuBvh.cu](./gpuBvh.cu)** - Defines a GPU version of the BVH that can be easily read and traversed in a CUDA kernel.
* **[gpuScene.cu](./gpuScene.cu)** - Defines a GPU version of the `Scene` structure.
* **[cudaHelpers.cu](./cudaHelpers.cu)** - Various helpers for working with CUDA code
* **[gpuTextures.cu](./gpuTextures.cu)** - Defines textures, loading textures from files, and sampling textures.
* **[kd-tree.cu](./kd-tree.cu)** - Implementation of a KD-tree for fast k-nearest-neighbors lookup on photon maps.
