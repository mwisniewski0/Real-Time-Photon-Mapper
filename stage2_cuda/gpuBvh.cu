#include "gpuBvh.h"


void BVHGpuData::release()
{
	triangles.release();
	bvhNodes.release();
}

BVHGpuData makeGpuBvh(const BVHNode* root)
{
	auto data = root->toRaw();

	BVHGpuData result;
	result.triangles = vectorToGpu(data->triangles);
	result.bvhNodes = vectorToGpu(data->bvhNodes);
	return result;
}
