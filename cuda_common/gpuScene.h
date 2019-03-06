#pragma once
#include "../cuda_common/cudaHelpers.h"
#include "../common/geometry.h"
#include "../common/scene.h"
#include "../cuda_common/gpuBvh.h"

/*
 * Keeps track of geometry and materials on the gpu.
 */
struct SceneInfo
{
	GPUVector<Sphere> spheres;
	GPUVector<PointLightSource> lights;
	GPUVector<Material> materials;
	BVHGpuData triangleBvh;

	// Builds a GPU scene from the provided CPU scene
	static SceneInfo fromScene(const Scene& scene);
};

// Converts the provided raw BVH data to a GPU compatible structure.
BVHGpuData rawBVHToGpu(const BVHGpuDataRaw& data);
