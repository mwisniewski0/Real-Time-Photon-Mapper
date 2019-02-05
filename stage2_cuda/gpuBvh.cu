#include "gpuBvh.h"

void makeGpuBvhInternal(const BVHNode* node, BVHGpuDataRaw* data)
{
	data->bvhNodes.emplace_back();
	int currIndex = data->bvhNodes.size() - 1;
	auto box = node->boundingBox;
	data->bvhNodes[currIndex].setBoundingBox(box);

	if (node->isLeaf())
	{
		const BVHLeaf* leaf = dynamic_cast<const BVHLeaf*>(node);
		data->bvhNodes[currIndex].setAsLeaf();
		data->bvhNodes[currIndex].setCount(leaf->triangles.size());
		data->bvhNodes[currIndex].u.leaf.offset = data->triangles.size();

		for (const auto& shape : leaf->triangles)
		{
			data->triangles.push_back(shape);
		}
	}
	else
	{
		const BVHInner* inner = dynamic_cast<const BVHInner*>(node);
		data->bvhNodes[currIndex].u.inner.left = data->bvhNodes.size(); // The next node will be the left node
		makeGpuBvhInternal(inner->left.get(), data);

		data->bvhNodes[currIndex].u.inner.right = data->bvhNodes.size(); // The next node will be the right node
		makeGpuBvhInternal(inner->right.get(), data);
	}
}

void GpuBvhNode::setBoundingBox(BoundingBox box)
{
	min = box.minCoords;
	max = box.maxCoords;
}

void BVHGpuData::release()
{
	triangles.release();
	bvhNodes.release();
}

BVHGpuData makeGpuBvh(const BVHNode* root)
{
	BVHGpuDataRaw data;
	makeGpuBvhInternal(root, &data);

	BVHGpuData result;
	result.triangles = vectorToGpu(data.triangles);
	result.bvhNodes = vectorToGpu(data.bvhNodes);
	return result;
}
