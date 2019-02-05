#pragma once

#include <float.h> // for max and min
#include <cuda_runtime.h>
#include <vector>
#include "geometry.h"
#include <iostream>
#include "cutil_math.h"
#include "cudaHelpers.h"
#include <memory>


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

	__device__ Triangle* intersectRay(const Ray& ray, float& out_distanceFromRayOrigin) const
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
				Box b;
				b.min = node.min;
				b.max = node.max;
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

// bvh interface on host
struct BVHNode{
	BoundingBox boundingBox;
    virtual bool isLeaf() = 0;
	BVHGpuData makeGpuBvh();
};


// intermediate node. points to left and right
struct BVHInner: BVHNode {
    std::unique_ptr<BVHNode> left;
    std::unique_ptr<BVHNode> right;
    virtual bool isLeaf(){return false;}
};

// leaf node in tree. contains list of triangles
struct BVHLeaf: BVHNode{
    std::vector<Triangle> triangles;
    virtual bool isLeaf(){return true;}
};

// triangles that haven't been added to bvh yet 
struct BBoxTemp{
    float3 min;
    float3 max;
    float3 center;
    Triangle triangle;
    BBoxTemp() :
	min({FLT_MAX, FLT_MAX, FLT_MAX}),
	max({-FLT_MAX, -FLT_MAX, -FLT_MAX})
    {}
};

extern std::vector<Triangle> triangles_ordered;
extern std::vector<GpuBvhNode> compact_BVH;

std::unique_ptr<BVHNode> buildBVH(std::vector<Triangle>&& triangles);
