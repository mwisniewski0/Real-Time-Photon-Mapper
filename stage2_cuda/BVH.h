#pragma once

#include <float.h> // for max and min
#include <cuda_runtime.h>
#include <vector>
#include "geometry.h"
#include <iostream>
#include "cutil_math.h"
#include "cudaHelpers.h"
#include <memory>

//this is all heavily based on http://raytracey.blogspot.com/2016/01/gpu-path-tracing-tutorial-3-take-your.html

// each node takes up 32 bytes to allign nicely in memory
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

	__device__ bool isLeaf();
	void setBoundingBox(BoundingBox box);
};

struct BVHGpuData
{
	GPUVector<GpuBvhNode> bvhNodes;
	GPUVector<Triangle> triangles;
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
