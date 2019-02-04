#include "geometry.h"
#include <vector>

__device__ __host__ float Triangle::intersect(const Ray& r) const
{
	float3 tvec = r.origin - p;
	float3 pvec = cross(r.dir, v1);
	float det = dot(v0, pvec);

	// TODO:
	det = 1.0 / det;

	float u = dot(tvec, pvec) * det;
	if (u < 0 || u > 1)
		return -1e20;

	float3 qvec = cross(tvec, v0);

	float v = dot(r.dir, qvec) * det;

	if (v < 0 || (u + v) > 1)
		return -1e20;

	return dot(v1, qvec) * det;
}

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

Triangle Triangle::from3Points(float3 v1, float3 v2, float3 v3, Material material)
{
	Triangle result;
	result.p = v1;
	result.v0 = v2 - v1;
	result.v1 = v3 - v1;
	result.material = material;
	result.normal = normalize(cross(result.v0, result.v1));
	return result;
}

BoundingBox Triangle::getBoundingBox() const
{
	BoundingBox result;
	return result.merge(p).merge(p + v0).merge(p + v1);
}

float Triangle::approxSurfaceArea() const
{
	return length(cross((v0), (v1))) * 0.5;
}

float3 Triangle::center() const
{
	return (3 * p + v0 + v1) / 3.f;
}
