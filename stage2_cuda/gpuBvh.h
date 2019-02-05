#pragma once

#include "../common/bvh.h"
#include "cudaHelpers.h"


// Representation of BVH nodes for GPUs. This is exactly 32 bytes which is the size of an L2 cache
// line.
struct GpuBvhNode {
	float3 min;
	float3 max;

	union {
		struct {
			unsigned left;
			unsigned right;
		} inner;
		struct {
			unsigned count;
			unsigned offset;
		} leaf;
	}u;

	__host__ __device__ bool isLeaf() const
	{
		return u.leaf.count & (1 << 31);
	}

	void setAsLeaf()
	{
		u.leaf.count |= (1 << 31);
	}

	void setCount(const unsigned& newCount)
	{
		u.leaf.count |= newCount;
	}

	__host__ __device__ unsigned getCount() const
	{
		return u.leaf.count & 0x7fffffff;
	}

	void setBoundingBox(BoundingBox box);
};

struct BVHGpuData
{
	GPUVector<GpuBvhNode> bvhNodes;
	GPUVector<Triangle> triangles;

	__host__ __device__ Triangle* intersectRay(const Ray& ray, float& out_distanceFromRayOrigin) const
	{
		Triangle* result = nullptr;

		// This should be sufficient for trees up to 32 levels deep (that is about 4 billion nodes)
		const int MAX_STACK_DEPTH = 64;

		int dfsStack[MAX_STACK_DEPTH];
		int stackSize = 1;
		dfsStack[0] = 0;
		
		while (stackSize)
		{
			// Popping from the stack
			const auto& node = bvhNodes.contents[dfsStack[stackSize - 1]];
			stackSize--;

			if (node.isLeaf())
			{
				for (int i = 0; i < node.getCount(); ++i)
				{
					auto& triangle = triangles.contents[i + node.u.leaf.offset];
					float distanceToCollision = triangle.intersect(ray);
					if (distanceToCollision > 0.001)  // If collision happened
					{
						if (distanceToCollision < out_distanceFromRayOrigin)
						{
							out_distanceFromRayOrigin = distanceToCollision;
							result = &triangle;
						}
					}
				}
			}
			else
			{
				BoundingBox b;
				b.minCoords = node.min;
				b.maxCoords = node.max;
				if (b.intersect(ray))
				{
					dfsStack[stackSize++] = node.u.inner.left; // push right and left onto stack
					dfsStack[stackSize++] = node.u.inner.right;
				}
			}
		}
		return result;
	}
};

BVHGpuData makeGpuBvh(const BVHNode* root);
