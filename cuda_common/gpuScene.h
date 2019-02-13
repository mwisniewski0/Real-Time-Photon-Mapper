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

	static SceneInfo fromScene(const Scene& scene);
};


BVHGpuData rawBVHToGpu(const BVHGpuDataRaw& data);
