#pragma once

#include "cutil_math.h"
#include <vector>
#include "geometry.h"
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

struct BVHGpuDataRaw
{
	std::vector<GpuBvhNode> bvhNodes;
	std::vector<Triangle> triangles;
};

// bvh interface on host
struct BVHNode{
	BoundingBox boundingBox;
    virtual bool isLeaf() const = 0;

	std::unique_ptr<BVHGpuDataRaw> toRaw() const;
};

// intermediate node. points to left and right
struct BVHInner: BVHNode {
    std::unique_ptr<BVHNode> left;
    std::unique_ptr<BVHNode> right;
    virtual bool isLeaf() const {return false;}
};

// leaf node in tree. contains list of triangles
struct BVHLeaf: BVHNode{
    std::vector<Triangle> triangles;
    virtual bool isLeaf() const {return true;}
};

std::unique_ptr<BVHNode> buildBVH(std::vector<Triangle>&& triangles);
