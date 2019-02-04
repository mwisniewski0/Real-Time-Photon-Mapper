#include "bvh.h"

std::vector<Triangle> triangles_ordered;
std::vector<GpuBvhNode> compact_BVH;


// recursively build bvh split on planes parallel to axes
// tries to keep surface area the same for both halves
std::unique_ptr<BVHNode> recurse(std::vector<BBoxTemp> working, int depth = 0) {
	if (working.size() < 4) { // if only 4 triangles left
		auto leaf = std::make_unique<BVHLeaf>();
		for (int i = 0; i< working.size(); ++i)
			leaf->triangles.push_back(working[i].triangle);
		return leaf;
	}
	float3 min = { FLT_MAX,FLT_MAX,FLT_MAX };
	float3 max = { -FLT_MAX,-FLT_MAX,-FLT_MAX };

	// calculate bounds for current working list
	for (unsigned i = 0; i<working.size(); ++i) {
		BBoxTemp& v = working[i];
		min = minf3(min, v.min);
		max = maxf3(max, v.max);
	}

	//approxomate SA of triangle by size of bounding box
	float side1 = max.x - min.x;
	float side2 = max.y - min.y;
	float side3 = max.z - min.z;

	float min_cost = working.size() * (side1*side2 +
		side2*side3 +
		side3*side1);
	float best_split = FLT_MAX; // best value along axis

	int best_axis = -1; // best axis

						// 0 = X-axis, 1 = Y-axis, 2=Z-axis
	for (int i = 0; i< 3; ++i) { //check all three axes
		int axis = i;
		float start, stop, step;
		if (axis == 0) {
			start = min.x;
			stop = max.x;
		}
		else if (axis == 1) {
			start = min.y;
			stop = max.y;
		}
		else {
			start = min.z;
			stop = max.z;
		}

		// if box is too thin in this dir
		if (fabsf(stop - start) < 1e-4)
			continue;

		// check discrete number of different splits on each axis
		// number gets smaller as we get farther into bvh and presumably smaller differences
		step = (stop - start) / (1024.0 / (depth + 1));

		// determine how good each plane is for splitting
		for (float test_split = start + step; test_split < stop - step; test_split += step) {
			float3 lmin = { FLT_MAX,FLT_MAX,FLT_MAX };
			float3 lmax = { -FLT_MAX,-FLT_MAX,-FLT_MAX };

			float3 rmin = { FLT_MAX,FLT_MAX,FLT_MAX };
			float3 rmax = { -FLT_MAX,-FLT_MAX,-FLT_MAX };

			int lcount = 0;
			int rcount = 0;

			for (unsigned j = 0; j<working.size(); ++j) {
				BBoxTemp& v = working[j];
				float val;
				// use triangle center to determine which side to put it in
				if (axis == 0) val = v.center.x;
				else if (axis == 1) val = v.center.y;
				else val = v.center.z;

				if (val < test_split) {
					lmin = minf3(lmin, v.min);
					lmax = maxf3(lmax, v.max);
					lcount++;
				}
				else {
					rmin = minf3(rmin, v.min);
					rmax = maxf3(rmax, v.max);
					rcount++;
				}
			}

			if (lcount <= 1 || rcount <= 1) continue;

			float lside1 = lmax.x - lmin.x;
			float lside2 = lmax.y - lmin.y;
			float lside3 = lmax.z - lmin.z;

			float rside1 = rmax.x - rmin.x;
			float rside2 = rmax.y - rmin.y;
			float rside3 = rmax.z - rmin.z;

			float lsurface = lside1*lside2 + lside2*lside3 + lside3*lside1;
			float rsurface = rside1*rside2 + rside2*rside3 + rside3*rside1;

			float total_cost = lsurface*lcount + rsurface*rcount;
			if (total_cost < min_cost) { // if this split is better, update stuff
				min_cost = total_cost;
				best_split = test_split;
				best_axis = axis;
			}
		}
	}
	// if no split is better, just add a leaf node
	if (best_axis == -1) {
		auto leaf = std::make_unique<BVHLeaf>();
		for (int i = 0; i< working.size(); ++i)
			leaf->triangles.push_back(working[i].triangle);
		return leaf;
	}

	// otherwise, create left and right working lists and call function recursively
	std::vector<BBoxTemp> left;
	std::vector<BBoxTemp> right;
	float3 lmin = { FLT_MAX,FLT_MAX,FLT_MAX };
	float3 lmax = { -FLT_MAX,-FLT_MAX,-FLT_MAX };
	float3 rmin = { FLT_MAX,FLT_MAX,FLT_MAX };
	float3 rmax = { -FLT_MAX,-FLT_MAX,-FLT_MAX };

	for (unsigned i = 0; i<working.size(); ++i) {
		BBoxTemp& v = working[i];
		float val;
		if (best_axis == 0) val = v.center.x;
		else if (best_axis == 1) val = v.center.y;
		else val = v.center.z;
		if (val < best_split) {
			left.push_back(v);
			lmin = minf3(lmin, v.min);
			lmax = maxf3(lmax, v.max);
		}
		else {
			right.push_back(v);
			rmin = minf3(rmin, v.min);
			rmax = maxf3(rmax, v.max);
		}
	}

	//create left and right child nodes
	auto inner = std::make_unique<BVHInner>();
	inner->left = recurse(left, depth + 1);
	inner->left->boundingBox.minCoords = lmin;
	inner->left->boundingBox.maxCoords = lmax;

	inner->right = recurse(right, depth + 1);
	inner->right->boundingBox.minCoords = rmin;
	inner->right->boundingBox.maxCoords = rmax;

	return inner;
}

struct BVHGpuDataRaw
{
	std::vector<GpuBvhNode> bvhNodes;
	std::vector<Triangle> triangles;

	BVHGpuData toGpu() const
	{
		BVHGpuData data;
		data.triangles = vectorToGpu(triangles);
		data.bvhNodes = vectorToGpu(bvhNodes);
		return data;
	}
};


void makeGpuBvhInternal(BVHNode* node, BVHGpuDataRaw* data)
{
	data->bvhNodes.emplace_back();
	int currIndex = data->bvhNodes.size() - 1;
	auto box = node->boundingBox;
	data->bvhNodes[currIndex].setBoundingBox(box);

	if (node->isLeaf())
	{
		BVHLeaf* leaf = dynamic_cast<BVHLeaf*>(node);
		// TODO: document 0x80000000
		data->bvhNodes[currIndex].u.leaf.count = 0x80000000 | leaf->triangles.size();
		data->bvhNodes[currIndex].u.leaf.offset = data->triangles.size();

		for (const auto& shape : leaf->triangles)
		{
			data->triangles.push_back(shape);
		}
	}
	else
	{
		BVHInner* inner = dynamic_cast<BVHInner*>(node);
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


BVHGpuData BVHNode::makeGpuBvh()
{
	BVHGpuDataRaw data;
	makeGpuBvhInternal(this, &data);
	return data.toGpu();
}

std::unique_ptr<BVHNode> buildBVH(std::vector<Triangle>&& triangles)
{
	std::vector<BBoxTemp> working;
	float3 min = {FLT_MAX, FLT_MAX, FLT_MAX};
	float3 max = {-FLT_MAX, -FLT_MAX, -FLT_MAX};

	std::cout << "Gathering box info..." << std::endl;

	for (unsigned i = 0; i < triangles.size(); ++i)
	{
		const Triangle& triangle = triangles[i];

		BBoxTemp b;
		b.triangle = triangle;

		b.min = minf3(b.min, triangle.p);
		b.min = minf3(b.min, triangle.p + triangle.v0);
		b.min = minf3(b.min, triangle.p + triangle.v1);

		b.max = maxf3(b.max, triangle.p);
		b.max = maxf3(b.max, triangle.p + triangle.v0);
		b.max = maxf3(b.max, triangle.p + triangle.v1);

		min = minf3(min, b.min);
		max = maxf3(max, b.max);

		b.center = (b.max + b.min) * 0.5;

		working.push_back(b);
	}

	std::cout << "Creating BVH..." << std::endl;

	auto root = recurse(working);
	root->boundingBox.minCoords = min;
	root->boundingBox.maxCoords = max;
	return root;
}
