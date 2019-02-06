#include "gpuScene.h"


SceneInfo SceneInfo::fromScene(const Scene& scene)
{
	SceneInfo result;

	std::vector<Triangle> cop = scene.triangles;
	std::unique_ptr<BVHNode> bvh = buildBVH(std::move(cop));
	result.triangleBvh = makeGpuBvh(bvh.get());

	result.spheres = vectorToGpu(scene.spheres);
	result.lights = vectorToGpu(scene.lights);

	return result;
}