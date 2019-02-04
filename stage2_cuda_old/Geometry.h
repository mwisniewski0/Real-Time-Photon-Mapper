#ifndef GEOMETRY_H_INCLUDED__
#define GEOMETRY_H_INCLUDED__
#include <vector>
#include <glm/vec3.hpp>
#include <GL/glew.h>
#include <memory>
#include <vector_types.h>
#include "helper_math.h"


// return min and max components of a vector
inline __host__ __device__ float3 minf3(float3 a, float3 b) {
	return make_float3(a.x<b.x ? a.x : b.x, a.y<b.y ? a.y : b.y, a.z<b.z ? a.z : b.z);
}
inline __host__ __device__ float3 maxf3(float3 a, float3 b) {
	return make_float3(a.x>b.x ? a.x : b.x, a.y>b.y ? a.y : b.y, a.z>b.z ? a.z : b.z);
}

struct BoundingBox
{
	float3 minCoords = make_float3(std::numeric_limits<float>::infinity());
	float3 maxCoords = make_float3(-std::numeric_limits<float>::infinity());

	BoundingBox merge(const BoundingBox& other) const;
	BoundingBox merge(const float3& v) const;
	float getArea();
};

class Shape
{
public:
	virtual BoundingBox getBoundingBox() const = 0;
	virtual float approxSurfaceArea() const = 0;
	virtual float3 center() const = 0;
};

struct vec4 {
	float x, y, z, w;
};

struct Material {
	float3 color;
	float3 specularReflectivity;
	float refractiveIndex;
	int type; // 0 diffuse, 1 specular, 2 refractive

	int padding1;
	int padding2;
};

struct Triangle : public Shape {
	float3 a, b, c;
	Material material;

	BoundingBox getBoundingBox() const override;
	float approxSurfaceArea() const override;
	float3 center() const override;
};

struct Sphere
{
	float3 center;
	float radius;
	Material material;
};

struct PointLightSource {
	float3 position;
	float3 intensity;
};

struct Scene
{
	std::vector<Sphere> spheres;
	std::vector<Triangle> triangles;
	std::vector<PointLightSource> lights;
};

BoundingBox getBoundingBoxForAllShapes(const std::vector<Triangle>& shapes);

#endif