#pragma once
#include "../cuda_common/cudaHelpers.h"
#include "../common/geometry.h"
#include "../common/scene.h"
#include "../cuda_common/gpuBvh.h"

struct SceneInfo
{
	GPUVector<Sphere> spheres;
	GPUVector<PointLightSource> lights;
	BVHGpuData triangleBvh;

	static SceneInfo fromScene(const Scene& scene);
};
