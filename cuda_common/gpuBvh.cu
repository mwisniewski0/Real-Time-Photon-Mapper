#include "../cuda_common/gpuBvh.h"


void BVHGpuData::release()
{
	triangles.release();
	bvhNodes.release();
}

BVHGpuData rawBVHToGpu(const BVHGpuDataRaw& data)
{
	BVHGpuData result;
	result.triangles = vectorToGpu(data.triangles);
	result.bvhNodes = vectorToGpu(data.bvhNodes);
	return result;
}

BVHGpuData makeGpuBvh(const BVHNode* root)
{
	auto data = root->toRaw();

	return rawBVHToGpu(*data);
}
