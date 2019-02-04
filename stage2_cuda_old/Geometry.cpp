#include "Geometry.h"
#include <memory>
#include <glm/glm.hpp>
#include "helper_math.h"

BoundingBox BoundingBox::merge(const BoundingBox& other) const
{
	BoundingBox result;
	result.minCoords = minf3(this->minCoords, other.minCoords);
	result.maxCoords = maxf3(this->maxCoords, other.maxCoords);
	return result;
}

BoundingBox BoundingBox::merge(const float3& v) const
{
	BoundingBox result;
	result.minCoords = minf3(this->minCoords, v);
	result.maxCoords = maxf3(this->maxCoords, v);
	return result;
}

float BoundingBox::getArea()
{
	return 2.f * dot((maxCoords - minCoords), (maxCoords - minCoords));
}

BoundingBox Triangle::getBoundingBox() const
{
	BoundingBox result;
	return result.merge(a).merge(b).merge(c);
}

float Triangle::approxSurfaceArea() const
{
	return length(cross((b - a), (c - a))) * 0.5;
}

float3 Triangle::center() const
{
	return (a + b + c) / 3.f;
}

BoundingBox getBoundingBoxForAllShapes(const std::vector<Triangle>& shapes)
{
	BoundingBox result; // Note that bounding boxes are initialized to an empty box
	for (const auto& shape : shapes)
	{
		result = result.merge(shape.getBoundingBox());
	}
	return result;
}
