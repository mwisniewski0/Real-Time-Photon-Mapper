#pragma once

// This file implements geometry data structures and functions, e.g. triangles, rays, bounding
// boxes, materials, as well as geometrical objects intersection, etc.

#include <vector>
#include <chrono>
#include "../common/cutil_math.h"
#include "streamWriter.h"
#include <sstream>

#ifdef NO_CUDA
#include "gpuTexturesNoCuda.h"
#else
#include "../cuda_common/gpuTextures.h"
#endif

// Represents a ray as defined here: https://en.wikipedia.org/wiki/Line_(geometry)#Ray
struct Ray {
	float3 origin;
	float3 dir;  // Unit dir vector

	// Creates a Ray from two points - the origin point, and any other point on the ray
	__device__ static Ray fromPoints(const float3& start, const float3& through)
	{
		Ray ray;
		ray.origin = start;
		ray.dir = normalize(through - start);
		return ray;
	}

	// Finds a point on the ray at the given distance from the origin
	__device__ float3 pointAtDistance(float distance) const
	{
		return origin + (dir * distance);
	}
};

// A bounding box is an axis-aligned 3D box. It is used for performing quick, approximate collisions
// between triangles and rays, and splitting triangles into a BVH tree
struct BoundingBox
{
	// Bottom-left corner of the box
	float3 minCoords = make_float3(std::numeric_limits<float>::infinity());

	// Top-right corner of the box
	float3 maxCoords = make_float3(-std::numeric_limits<float>::infinity());

	// Returns the smallest BoundingBox that contains both this and the other bounding box.
	BoundingBox merge(const BoundingBox& other) const;

	// Returns the smallest BoundingBox that contains this bounding box and the provided point.
	BoundingBox merge(const float3& v) const;

	// Returns the area of the box's surface
	float getArea();

	// Intersects this bounding box with the provided ray. If the ray does not intersect the box,
	// a negative number is returned. Otherwise, the distance from the ray origin to the point of
	// intersection is returned.
	// TODO: possible bug with the returned value. Verify the usages as well
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

// Describes the material out of which a given object within the scene is made. Refer to the MTL
// file spec for more information about each field
// (https://en.wikipedia.org/wiki/Wavefront_.obj_file)
struct Material {
	// How much of the incident light should be reflected diffusely (this is component-wise
	// with the float3 storing values in range [0, 1] for the R, G, B channels - in this order).
	float3 diffuse;

	// How much of the incident light should be reflected specularly (this is component-wise
	// with the float3 storing values in range [0, 1] for the R, G, B channels - in this order).
	float3 specular;

	// How much of the incident light should be transmitted into the object (this is component-wise
	// with the float3 storing values in range [0, 1] for the R, G, B channels - in this order).
	float3 transmittance;

	// Defines how blurry the specular reflection is, with 0 being completely diffuse and 1 being
	// perfectly specular.
	float shininess; // This is currently ignored and assumed to be 1.0

	// The refractive index of the material as used in Snell's law
	float refractiveIndex;

	// Specifies whether a diffuse texture should be used instead of a constant diffuse reflection
	// value.
	bool useDiffuseTexture;

	// If useDiffuseTexture is set to true, this specifies the diffuse texture of the material.
	// Otherwise, this field is ignored.
	GPUTexture diffuseTexture;
};

// Simple wrapper for the Material structure that stores the path to the diffuse texture and can
// therefore be saved to/loaded from the hard drive.
struct MaterialInfo {
	// This stores information about the material; note that material.diffuseTexture is ignored.
	Material material;

	// Path to the diffuse texture to be used by this material. This will be ignored if
	// material.useDiffuseTexture is set to false
	std::string diffuseTexturePath;

	// Returns a Material object with a loaded diffuse texture dependent on the diffuseTexturePath.
	// Such Material object is ready to be passed to the GPU and used as needed.
	Material loadWithTexture() const;

	// Returns textual representation of this object for testing purposes.
	__host__ std::string toString()
	{
		std::stringstream s;
		s << "(Diffuse: " << float3ToString(material.diffuse) << ", ";
		s << "Specular: " << float3ToString(material.specular) << ", ";
		s << "Transmittance: " << float3ToString(material.transmittance) << ", ";
		s << "Shininess: " << material.shininess << ", ";
		s << "Refractive Index: " << material.refractiveIndex << ", ";
		s << "Diffuse texture: " << diffuseTexturePath << ")";
		return s.str();
	}
};

// Specialization of the stream writer for MaterialInfo. Refer to streamWriter.h for more
// information.
template<>
void writeToStream<MaterialInfo>(std::ostream& s, const MaterialInfo& v);

// Specialization of the stream reader for MaterialInfo. Refer to streamWriter.h for more
// information.
template<>
MaterialInfo readFromStream<MaterialInfo>(std::istream& s);

// Represents a 3D sphere
struct Sphere {
	float3 center;
	float radius;
	Material material;

	// Intersects this sphere with the given ray. Returns a non-zero value if an intersection
	// happened. Saves the distance to the intersection from the ray origin in the provided dist
	// variable.
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


// Represents a Triangle in 3-space. The vertices are in counter-clockwise order. Instead of storing
// all 3 vertices of the triangle, we store one of the vertices and the vectors to the other ones.
// This allows us to skip a few computations during each ray-triangle intersection.
struct Triangle {
	// v0 vertex of the triangle
	float3 v0;

	// Vector from v0 to the v1 vertex of the triangle
	float3 v0v1;

	// Vector from v0 to the v2 vertex of the triangle
	float3 v0v2;

	float3 normal;

	// The index of the material used by this triangle as stored in the vector of materials in the
	// scene object.
	unsigned materialIndex;

	// Texture coordinates of each vertex. Note that the Z-part of these will be ignored
	float3 v0vt;
	float3 v1vt;
	float3 v2vt;

	// Normals for each vertex. This will be interpolated using barycentric coordinates to get an
	// estimate for curved surfaces.
	float3 v0vn;
	float3 v1vn;
	float3 v2vn;

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

	// Given a vector of triangles, returns the smallest bounding box containing all of the
	// triangles.
	static BoundingBox boundingBoxForMany(const std::vector<Triangle>& triangles);

	// Creates a triangle from 3 points
	static Triangle from3Points(float3 v1, float3 v2, float3 v3, unsigned materialIndex);

	// Returns the bounding box of this triangle
	BoundingBox getBoundingBox() const;

	// Returns the surface area of this triangle
	float surfaceArea() const;

	// Returns the center of mass of this triangle given uniform weight distribution.
	float3 center() const;

	// Returns a textual representation of this triangle for testing purposes.
	__host__ std::string toString()
	{
		std::stringstream s;
		s << "(v0: " << float3ToString(v0) << ", ";
		s << "v1: " << float3ToString(v0 + v0v1) << ", ";
		s << "v2: " << float3ToString(v0 + v0v2) << ", ";
		s << "v0vt: " << float3ToString(v0vt) << ", ";
		s << "v1vt: " << float3ToString(v1vt) << ", ";
		s << "v2vt: " << float3ToString(v2vt) << ", ";
		s << "v0vn: " << float3ToString(v0vn) << ", ";
		s << "v1vn: " << float3ToString(v1vn) << ", ";
		s << "v2vn: " << float3ToString(v2vn) << ", ";
		s << "normal: " << float3ToString(normal) << ", ";
		s << "material index: " << materialIndex << ")";
		return s.str();
	}
};

// Specialization of the stream writer for Triangles. Refer to streamWriter.h for more information.
template<>
void writeToStream<Triangle>(std::ostream& s, const Triangle& v);

// Specialization of the stream reader for Triangles. Refer to streamWriter.h for more information.
template<>
Triangle readFromStream<Triangle>(std::istream& s);

// Represents a simple point light source. The intensity is given as a vector of 3 floating point
// values with 0 being no light and 1 being full intensity. The values come in the order of R, G, B.
struct PointLightSource {
	float3 position;
	float3 intensity;
};

// Specialization of the stream writer for PointLightSources. Refer to streamWriter.h for more
// information.
template<>
void writeToStream<PointLightSource>(std::ostream& s, const PointLightSource& v);

// Specialization of the stream reader for PointLightSources. Refer to streamWriter.h for more
// information.
template<>
PointLightSource readFromStream<PointLightSource>(std::istream& s);

// Returns the area of a triangle given its 3 points.
inline float triangleArea(const float3& v0, const float3& v1, const float3& v2)
{
	return length(cross((v1 - v0), (v2 - v0))) * 0.5;
}

// Converts a point p in absolute coordinates to the barycentric coordinates of the given triangle.
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

// Applies the given barycentric coordinates on a new triangle with vertices a, b, c. Returns the
// absolute coordinates of the resultant point.
inline __device__ __host__ float3 applyBarycentric(const float3& bary, const float3& a, const float3& b, const float3& c)
{
	return bary.x * a + bary.y * b + bary.z * c;
}