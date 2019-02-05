#pragma once
#define M_PI   3.14159265358979323846264338327950288

#include <vector>
#include <chrono>
#include "../common/cutil_math.h"

struct BoundingBox
{
	float3 minCoords = make_float3(std::numeric_limits<float>::infinity());
	float3 maxCoords = make_float3(-std::numeric_limits<float>::infinity());

	BoundingBox merge(const BoundingBox& other) const;
	BoundingBox merge(const float3& v) const;
	float getArea();
};

struct Material {
	float3 color;
	float3 specularReflectivity;
	float refractiveIndex;
	int type; // 0 diffuse, 1 specular, 2 refractive
};

class Shape
{
public:
	virtual BoundingBox getBoundingBox() const = 0;
	virtual float approxSurfaceArea() const = 0;
	virtual float3 center() const = 0;
};

struct Ray {
	float3 origin;
	float3 dir;  // Unit dir vector

	__device__ static Ray fromPoints(const float3& start, const float3& through)
	{
		Ray ray;
		ray.origin = start;
		ray.dir = normalize(through - start);
		return ray;
	}

	__device__ float3 pointAtDistance(float distance) const
	{
		return origin + (dir * distance);
	}
};

// TODO: only used as a bounding box, remove
struct Box {
	float3 max;
	float3 min;

	__device__ float intersect(const Ray& r) const {
		if (min.x < r.origin.x && r.origin.x < max.x &&
			min.y < r.origin.y && r.origin.y < max.y &&
			min.z < r.origin.z && r.origin.z < max.z) return -1;

		float3 tmin = (min - r.origin) / r.dir;
		float3 tmax = (max - r.origin) / r.dir;

		float3 rmin = minf3(tmin, tmax);
		float3 rmax = maxf3(tmin, tmax);

		float minmax = minf(minf(rmax.x, rmax.y), rmax.z);
		float maxmin = maxf(maxf(rmin.x, rmin.y), rmin.z);

		if (minmax >= maxmin) return maxmin > 0.000001 ? maxmin : 0;
		else return 0;
	}
};

struct Sphere {
	float3 center;
	float radius;
	Material material;

	// use quadratic formula
	// if one positive solution, started inside sphere
	// difference of solutions in the distance between to two sides of the sphere
	inline __device__ float intersect(const Ray& r, float& dist) const {
		float3 to_origin = center - r.origin;
		float b = dot(to_origin, r.dir);
		float c = dot(to_origin, to_origin) - radius*radius;
		float disc = b*b - c;
		if (disc>0) {
			disc = sqrtf(disc);
			float t = b - disc;
			if (t>0.0001) {
				dist = 1e20;
				return t;
			}
			t = b + disc;
			if (t>0.0001) {
				dist = 2 * disc;
				return t;
			}
		}
		return 0;
	}
};


// data for a single triangle
// currently has sepparate scatter and texture info for each triangle
// scatter and texture type and texture num should eventually be moved to mesh
struct Triangle : public Shape {
	float3 p;
	float3 v0;
	float3 v1;
	float3 normal; //precompute and store. may not be faster needs testing
	Material material;

	// Moller-Trumbore algorithm for triangle-ray intersection. Returns < 0 if no intersection
	// occurred. If intersection occured the result will be the distance of the intersection point
	// to the ray origin
	__device__ float intersect(const Ray& r) const
	{
		float3 tvec = r.origin - p;
		float3 pvec = cross(r.dir, v1);
		float det = dot(v0, pvec);

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

	static BoundingBox boundingBoxForMany(const std::vector<Triangle>& triangles);

	static Triangle from3Points(float3 v1, float3 v2, float3 v3, Material material);

	BoundingBox getBoundingBox() const override;
	float approxSurfaceArea() const override;
	float3 center() const override;
};

struct PointLightSource {
	float3 position;
	float3 intensity;
};
