#pragma once


#include <vector>
#include <chrono>
#include "../common/cutil_math.h"
#include "streamWriter.h"

#ifdef NO_CUDA
#include "gpuTexturesNoCuda.h"
#else
#include "../cuda_common/gpuTextures.h"
#endif

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

struct BoundingBox
{
	float3 minCoords = make_float3(std::numeric_limits<float>::infinity());
	float3 maxCoords = make_float3(-std::numeric_limits<float>::infinity());

	BoundingBox merge(const BoundingBox& other) const;
	BoundingBox merge(const float3& v) const;
	float getArea();

	__device__ __host__ float intersect(const Ray& r) const {
		if (minCoords.x < r.origin.x && r.origin.x < maxCoords.x &&
			minCoords.y < r.origin.y && r.origin.y < maxCoords.y &&
			minCoords.z < r.origin.z && r.origin.z < maxCoords.z) return -1;

		float3 tmin = (minCoords - r.origin) / r.dir;
		float3 tmax = (maxCoords - r.origin) / r.dir;

		float3 rmin = minf3(tmin, tmax);
		float3 rmax = maxf3(tmin, tmax);

		float minmax = minf(minf(rmax.x, rmax.y), rmax.z);
		float maxmin = maxf(maxf(rmin.x, rmin.y), rmin.z);

		if (minmax >= maxmin) return maxmin > 0.000001 ? maxmin : 0;
		else return 0;
	}
};

struct Material {
	float3 diffuse;
	float3 specular;
	float3 transmittance;
	float shininess; // This is currently ignored and assumed to be 1.0
	float refractiveIndex;

	bool useDiffuseTexture; // checks whether to multiply diffuse by the texture
	GPUTexture diffuseTexture;
};

struct MaterialInfo {
	Material material;
	std::string diffuseTexturePath;

	Material loadWithTexture() const;
};

template<>
void writeToStream<MaterialInfo>(std::ostream& s, const MaterialInfo& v);

template<>
MaterialInfo readFromStream<MaterialInfo>(std::istream& s);

class Shape
{
public:
	virtual BoundingBox getBoundingBox() const = 0;
	virtual float approxSurfaceArea() const = 0;
	virtual float3 center() const = 0;
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
	float3 v0;
	float3 v0v1;
	float3 v0v2;
	float3 normal; //precompute and store. may not be faster needs testing

	unsigned materialIndex;

	// Z-part of these will be ignored
	float3 v0vt;
	float3 v1vt;
	float3 v2vt;

	// Moller-Trumbore algorithm for triangle-ray intersection. Returns < 0 if no intersection
	// occurred. If intersection occured the result will be the distance of the intersection point
	// to the ray origin
	__host__ __device__ float intersect(const Ray& r) const
	{
		float3 tvec = r.origin - v0;
		float3 pvec = cross(r.dir, v0v2);
		float det = dot(v0v1, pvec);

		det = 1.0 / det;

		float u = dot(tvec, pvec) * det;
		if (u < 0 || u > 1)
			return -1e20;

		float3 qvec = cross(tvec, v0v1);

		float v = dot(r.dir, qvec) * det;

		if (v < 0 || (u + v) > 1)
			return -1e20;

		return dot(v0v2, qvec) * det;
	}

	static BoundingBox boundingBoxForMany(const std::vector<Triangle>& triangles);

	static Triangle from3Points(float3 v1, float3 v2, float3 v3, unsigned materialIndex);

	BoundingBox getBoundingBox() const override;
	float approxSurfaceArea() const override;
	float3 center() const override;
};

template<>
void writeToStream<Triangle>(std::ostream& s, const Triangle& v);

template<>
Triangle readFromStream<Triangle>(std::istream& s);

struct PointLightSource {
	float3 position;
	float3 intensity;
};

template<>
void writeToStream<PointLightSource>(std::ostream& s, const PointLightSource& v);

template<>
PointLightSource readFromStream<PointLightSource>(std::istream& s);


inline float triangleArea(const float3& v0, const float3& v1, const float3& v2)
{
	return length(cross((v1 - v0), (v2 - v0))) * 0.5;
}

inline __device__ __host__ float3 absoluteToBarycentric(const Triangle& t, const float3& p)
{
	float3 result;

	const float3& a = t.v0;
	float3 b = t.v0v1 + a;
	float3 c = t.v0v2 + a;

	float abcArea = triangleArea(a, b, c);
	float capArea = triangleArea(c, a, p);
	float bcpArea = triangleArea(b, c, p);

	// contribution of v0
	result.x = bcpArea / abcArea;

	// contribution of v1
	result.y = capArea / abcArea;

	// contribution of v2
	result.z = 1 - result.x - result.y;

	return result;
}

inline __device__ __host__ float3 applyBarycentric(const float3& bary, const float3& a, const float3& b, const float3& c)
{
	return bary.x * a + bary.y * b + bary.z * c;
}