#pragma once

#include "cutil_math.h"
#include <vector>
#include "geometry.h"
#include <memory>

// bvh interface on host
struct BVHNode{
	BoundingBox boundingBox;
    virtual bool isLeaf() const = 0;
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
