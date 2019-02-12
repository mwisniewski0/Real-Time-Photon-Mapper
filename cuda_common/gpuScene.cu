#include "gpuScene.h"


SceneInfo SceneInfo::fromScene(const Scene& scene)
{
	SceneInfo result;

	result.triangleBvh = rawBVHToGpu(scene.triangleData);

	std::vector<Material> materials;
	for (const auto& matInfo : scene.materials)
	{
		materials.push_back(matInfo.loadWithTexture());
	}
	result.materials = vectorToGpu(materials);

	result.spheres = vectorToGpu(scene.spheres);
	result.lights = vectorToGpu(scene.lights);

	return result;
}