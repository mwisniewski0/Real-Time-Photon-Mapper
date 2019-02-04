#ifndef BVH_H_INCLUDED__
#define BVH_H_INCLUDED__
#include <glm/vec3.hpp>
#include <memory>
#include <vector>
#include "Geometry.h"
#include <tuple>
#include <stack>

// Based on	https://github.com/samkottler/GPU-Path-Tracer/blob/master/src/bvh.h

struct BVHNodeGlslFormat {
	float boundingBoxMinX;
	float boundingBoxMinY;
	float boundingBoxMinZ;
	int leftNodeOrObjectCount;
	float boundingBoxMaxX;
	float boundingBoxMaxY;
	float boundingBoxMaxZ;
	int rightNodeOrObjectOffset;

	void setBoundingBox(const BoundingBox& boundingBox);
	void setLeaf();
};



struct BVHGpuData
{
	std::vector<BVHNodeGlslFormat> bvhNodes;
	std::vector<Triangle> shapes;
};

// Bounding volume hierarchy nodes: inner and leaves.
struct BVHNode {
	BoundingBox boundingBox;
	virtual bool isLeaf() = 0;
	virtual ~BVHNode() = default;

	BVHGpuData makeGpuBvh();
};

struct BVHInner : BVHNode {
	std::unique_ptr<BVHNode> left;
	std::unique_ptr<BVHNode> right;
	bool isLeaf() override { return false; }
};

struct BVHLeaf : BVHNode {
	std::vector<Triangle> shapes;
	bool isLeaf() override { return true; }
};

std::unique_ptr<BVHNode> buildBVH(std::vector<Triangle>&& shapes);

#endif