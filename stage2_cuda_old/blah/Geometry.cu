#include "Geometry.h"
#include <memory>
#include <glm/glm.hpp>

BoundingBox BoundingBox::merge(const BoundingBox& other) const
{
	BoundingBox result;
	result.minCoords = glm::min(this->minCoords, other.minCoords);
	result.maxCoords = glm::max(this->maxCoords, other.maxCoords);
	return result;
}

BoundingBox BoundingBox::merge(const glm::vec3& v) const
{
	BoundingBox result;
	result.minCoords = glm::min(this->minCoords, v);
	result.maxCoords = glm::max(this->maxCoords, v);
	return result;
}

float BoundingBox::getArea()
{
	return 2.f * glm::dot((maxCoords - minCoords), (maxCoords - minCoords));
}

BoundingBox Triangle::getBoundingBox() const
{
	BoundingBox result;
	return result.merge(a).merge(b).merge(c);
}

float Triangle::approxSurfaceArea() const
{
	return glm::length(glm::cross((b - a), (c - a))) * 0.5;
}

glm::vec3 Triangle::center() const
{
	return (a + b + c) / 3.f;
}

GPUFriendlyTriangle Triangle::getGPUFriendly() const
{
	GPUFriendlyTriangle result;

	result.a.x = a.x;
	result.a.y = a.y;
	result.a.z = a.z;
	result.a.w = 0.f;

	result.b.x = b.x;
	result.b.y = b.y;
	result.b.z = b.z;
	result.b.w = 0.f;

	result.c.x = c.x;
	result.c.y = c.y;
	result.c.z = c.z;
	result.c.w = 0.f;
	
	result.material = material;
	
	return result;
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
