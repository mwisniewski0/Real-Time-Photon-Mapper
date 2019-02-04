#include "BVH.h"

void BVHNodeGpuFormat::setBoundingBox(const BoundingBox& boundingBox)
{
	this->boundingBoxMaxX = boundingBox.maxCoords.x;
	this->boundingBoxMaxY = boundingBox.maxCoords.y;
	this->boundingBoxMaxZ = boundingBox.maxCoords.z;

	this->boundingBoxMinX = boundingBox.minCoords.x;
	this->boundingBoxMinY = boundingBox.minCoords.y;
	this->boundingBoxMinZ = boundingBox.minCoords.z;
}

void BVHNodeGpuFormat::setLeaf()
{
	leftNodeOrObjectCount |= 1 << 31;
}

void makeGpuBvhInternal(BVHNode* node, BVHGpuData* data)
{
	data->bvhNodes.emplace_back();
	int currIndex = data->bvhNodes.size() - 1;
	data->bvhNodes[currIndex].setBoundingBox(node->boundingBox);

	if (node->isLeaf())
	{
		BVHLeaf* leaf = dynamic_cast<BVHLeaf*>(node);
		data->bvhNodes[currIndex].leftNodeOrObjectCount = leaf->shapes.size();
		data->bvhNodes[currIndex].setLeaf();
		data->bvhNodes[currIndex].rightNodeOrObjectOffset = data->shapes.size();

		for (const auto& shape : leaf->shapes)
		{
			data->shapes.push_back(shape.getGPUFriendly());
		}
	}
	else
	{
		BVHInner* inner = dynamic_cast<BVHInner*>(node);
		data->bvhNodes[currIndex].leftNodeOrObjectCount = data->bvhNodes.size(); // The next node will be the left node
		makeGpuBvhInternal(inner->left.get(), data);

		data->bvhNodes[currIndex].rightNodeOrObjectOffset = data->bvhNodes.size(); // The next node will be the right node
		makeGpuBvhInternal(inner->right.get(), data);
	}
}

BVHGpuData BVHNode::makeGpuBvh()
{
	BVHGpuData data;
	makeGpuBvhInternal(this, &data);
	return data;
}

unsigned const MAX_SHAPES_IN_LEAF = 4;

enum Axis
{
	AxisX = 1,
	AxisY = 2,
	AxisZ = 3
};

float getSplitCost(const std::vector<Triangle>& shapes, Axis splitAxis, float splitPosition)
{
	float leftArea = 0.0f;
	float rightArea = 0.0f;

	unsigned leftCount = 0;
	unsigned rightCount = 0;

	BoundingBox leftBox;
	BoundingBox rightBox;

	for (const auto& shape : shapes) {
		// Split between halves based on the center
		float position;
		switch (splitAxis)
		{
		case AxisX:
			position = shape.center().x;
			break;
		case AxisY:
			position = shape.center().y;
			break;
		case AxisZ:
			position = shape.center().z;
			break;
		}

		if (position < splitPosition) {
			leftArea += shape.approxSurfaceArea();
			leftBox = leftBox.merge(shape.getBoundingBox());
			leftCount++;
		}
		else {
			rightArea += shape.approxSurfaceArea();
			rightBox = rightBox.merge(shape.getBoundingBox());
			rightCount++;
		}
	}

    return leftBox.getArea()*leftCount + rightBox.getArea()*rightCount;
//	return leftArea*leftCount + rightArea*rightCount;
}

std::tuple<float, float> findBestSplit(const std::vector<Triangle>& shapes,
	Axis splitAxis, float minOnAxis, float maxOnAxis, int numOfTests)
{
	float step = (maxOnAxis - minOnAxis) / (numOfTests - 1);
	float currentSplit = minOnAxis;
	float lowestCost = std::numeric_limits<float>::infinity();
	float lowestCostSplit;

	for (unsigned i = 0; i < numOfTests; ++i, currentSplit += step)
	{
		auto cost = getSplitCost(shapes, splitAxis, currentSplit);
		if (cost < lowestCost)
		{
			lowestCost = cost;
			lowestCostSplit = currentSplit;
		}
	}
	return { lowestCostSplit, lowestCost };
}

std::unique_ptr<BVHNode> buildBVH(std::vector<Triangle>&& shapes) {
	auto boundingBox = getBoundingBoxForAllShapes(shapes);
	if (shapes.size() < MAX_SHAPES_IN_LEAF)
	{
		auto leaf = std::make_unique<BVHLeaf>();
		leaf->shapes = std::move(shapes);
		leaf->boundingBox = boundingBox;
		return leaf;
	}

	float costWithNoSplit = getSplitCost(shapes, AxisX, std::numeric_limits<float>::infinity());
	bool usefulSplitFound = false;

	float currentLowestCost = costWithNoSplit;
	Axis bestSplitAxis;
	float bestSplitCoord;

	// Try splitting on all axes
	for (auto axis : { AxisX, AxisY, AxisZ }) {
		float start, stop;
		switch (axis)
		{
		case AxisX:
			start = boundingBox.minCoords.x;
			stop = boundingBox.maxCoords.x;
			break;
		case AxisY:
			start = boundingBox.minCoords.y;
			stop = boundingBox.maxCoords.y;
			break;
		case AxisZ:
			start = boundingBox.minCoords.z;
			stop = boundingBox.maxCoords.z;
			break;
		}

		// Don't use this split if the box is too thin in this direction
		if (fabsf(stop - start) < 1e-4)
			continue;

		const int NUM_OF_TESTS = 1024;
		auto best = findBestSplit(shapes, axis, start, stop, NUM_OF_TESTS);
		auto bestSplit = std::get<0>(best);
		auto bestSplitCost = std::get<1>(best);

		if (bestSplitCost < currentLowestCost)
		{
			usefulSplitFound = true;
			bestSplitAxis = axis;
			bestSplitCoord = bestSplit;
		}
	}

	// if no split is better, just add a leaf node
	if (!usefulSplitFound) {
		auto leaf = std::make_unique<BVHLeaf>();
		leaf->shapes = std::move(shapes);
		leaf->boundingBox = boundingBox;
		return leaf;
	}

	// otherwise, create an interior node
	auto result = std::make_unique<BVHInner>();
	result->boundingBox = boundingBox;
	std::vector<Triangle> shapesLeft;
	std::vector<Triangle> shapesRight;

	for (const auto& shape : shapes)
	{
		float shapeCenterCoord;
		switch (bestSplitAxis)
		{
		case AxisX: shapeCenterCoord = shape.center().x; break;
		case AxisY: shapeCenterCoord = shape.center().y; break;
		case AxisZ: shapeCenterCoord = shape.center().z; break;
		}

		if (shapeCenterCoord < bestSplitCoord)
		{
			shapesLeft.push_back(std::move(shape));
		}
		else
		{
			shapesRight.push_back(std::move(shape));
		}
	}

	result->left = buildBVH(std::move(shapesLeft));
	result->right = buildBVH(std::move(shapesRight));
	return result;
}
