// This file implements a Bounding Volume Hierarchy data structure for O(log(n)) geometry search
// (with n being the number of triangles in the scene).
// See more about BVHs here: https://en.wikipedia.org/wiki/Bounding_volume_hierarchy

#pragma once

#include "cutil_math.h"
#include <vector>
#include "geometry.h"
#include <memory>
#include "streamWriter.h"

// Representation of BVH nodes for GPUs. This is exactly 32 bytes which is the size of an L2 cache
// line. These nodes will be stored in a contiguous chunk of memory, one after another. See
// BVHGpuDataRaw for specifics about the representation of the BVH in the GPU memory.
struct GpuBvhNode {
	// Minimum and maximum coordinates of the bounding box of this node
	float3 min;
	float3 max;

	// This union stores the data of the node. The node can be either an inner or a leaf node. The
	// node is considered a leaf if the most significant bit of u.leaf.count/u.inner.left is set.
	// If the bit is not set, the node is an inner node.
	// Use isLeaf(), setAsLeaf(), setCount() and getCount() to properly manage the type indicator
	// bit.
	union {
		// Information about the inner node - the index of the left and right sub-node.
		struct {
			unsigned left;
			unsigned right;
		} inner;

		// Information about the leaf node. This specifies the triangles that belong to this node.
		// Offset specifies the index of the first triangle in the list of triangles that belongs to
		// this node, whereas count specifies the number of triangles in this node. All triangles
		// with indices offset, offset+1, ..., offset+(count-1) are included in the node. Note that
		// if the node is a leaf node, the most significant bit of u.leaf.count is ignored when
		// trying to find the number of triangles in the node.
		struct {
			unsigned count;
			unsigned offset;
		} leaf;
	} u;

	// Checks the type indicator bit to check whether this is a leaf node. Returns true if this is a
	// leaf and false if this is an inner node.
	__host__ __device__ bool isLeaf() const
	{
	    return u.leaf.count & (uint)(1 << 31);
	}

	// Set the type indicator bit to specify that this is a leaf node.
	void setAsLeaf()
	{
	    u.leaf.count |= (uint)(1 << 31);
	}

	// Sets the u.leaf.count value while keeping the type indicator bit
	void setCount(const unsigned& newCount)
	{
		u.leaf.count |= newCount;
	}

	// Gets the u.leaf.count value while ignoring the type indicator bit
	__host__ __device__ unsigned getCount() const
	{
		return u.leaf.count & 0x7fffffff;
	}

	// Sets the bounding box of this node based on the BoundingBox structure provided in the
	// geometry module.
	void setBoundingBox(const BoundingBox& box);

	// Returns a textual representation of this node for debugging purposes.
	__host__ std::string toString();
};

// A representation of the BVH that is contiguous in memory. This allows for easy saving and loading
// from files, as well as easier transfer to the GPU memory and better memory locality.
struct BVHGpuDataRaw
{
	std::vector<GpuBvhNode> bvhNodes;
	std::vector<Triangle> triangles;
};

// Representation of a BVH node on the CPU machine. Note this is an abstract class - the BVHNode can
// be either an inner or a leaf node.
struct BVHNode{
	// The smallest bounding box containing all of the triangles in this node
	BoundingBox boundingBox;

	// Specifies whether this is a leaf node. This will return true on leaf nodes and false on inner
	// nodes.
    virtual bool isLeaf() const = 0;

	// Returns the contents of this node (including the sub-nodes) as raw GPU data. Data formatted
	// in this way can be easily and efficiently transported to and used on the GPU.
	std::unique_ptr<BVHGpuDataRaw> toRaw() const;
};

// An inner BVH node. Contains pointers to the left and right sub-nodes.
struct BVHInner: BVHNode {
    std::unique_ptr<BVHNode> left;
    std::unique_ptr<BVHNode> right;
    virtual bool isLeaf() const {return false;}
};

// A leaf BVH node. Contains a list of triangles.
struct BVHLeaf: BVHNode{
    std::vector<Triangle> triangles;
    virtual bool isLeaf() const {return true;}
};

// Builds a BVH from the provided vector of triangles. Note this is an expensive operation and might
// take a significant amount of time.
std::unique_ptr<BVHNode> buildBVH(std::vector<Triangle>&& triangles);

// Specialization of the stream writer for GpuBvhNodes. Refer to streamWriter.h for more
// information.
template<>
void writeToStream<GpuBvhNode>(std::ostream& s, const GpuBvhNode& v);

// Specialization of the stream reader for GpuBvhNodes. Refer to streamWriter.h for more
// information.
template<>
GpuBvhNode readFromStream<GpuBvhNode>(std::istream& s);

// Specialization of the stream writer for BVHGpuDataRaw. Refer to streamWriter.h for more
// information.
template<>
void writeToStream<BVHGpuDataRaw>(std::ostream& s, const BVHGpuDataRaw& v);

// Specialization of the stream reader for BVHGpuDataRaw. Refer to streamWriter.h for more
// information.
template<>
BVHGpuDataRaw readFromStream<BVHGpuDataRaw>(std::istream& s);
