#pragma once

#include "../common/bvh.h"
#include "../cuda_common/cudaHelpers.h"

/*
 * Keeps track of triangles and intersects bvh.
 */
struct BVHGpuData
{
	GPUVector<GpuBvhNode> bvhNodes;
	GPUVector<Triangle> triangles;
    
    /*
     * Uses the BVH to intersect a ray and the triangles. Returns the intersected triangle if there is 
     * one. Otherwise return nullptr. Because of the BVH this should be something like average O(logn)
     */
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

	void release();
};

BVHGpuData makeGpuBvh(const BVHNode* root);
