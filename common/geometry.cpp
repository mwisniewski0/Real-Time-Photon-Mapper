#include "geometry.h"
#include <vector>

BoundingBox Triangle::boundingBoxForMany(const std::vector<Triangle>& triangles)
{
	BoundingBox result;
	for (const auto& t : triangles)
	{
		result = result.merge(t.getBoundingBox());
	}
	return result;
}

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
	return dot((maxCoords - minCoords), (maxCoords - minCoords));
}

Triangle Triangle::from3Points(float3 v1, float3 v2, float3 v3, Material material)
{
	Triangle result;
	result.v0 = v1;
	result.v0v1 = v2 - v1;
	result.v0v2 = v3 - v1;
	result.material = material;
	result.normal = normalize(cross(result.v0v1, result.v0v2));
	return result;
}

BoundingBox Triangle::getBoundingBox() const
{
	BoundingBox result;
	return result.merge(v0).merge(v0 + v0v1).merge(v0 + v0v2);
}

float Triangle::approxSurfaceArea() const
{
	return length(cross((v0v1), (v0v2))) * 0.5;
}

float3 Triangle::center() const
{
	return (3 * v0 + v0v1 + v0v2) / 3.f;
}
