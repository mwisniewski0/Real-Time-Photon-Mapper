#ifndef GEOMETRY_H_INCLUDED__
#define GEOMETRY_H_INCLUDED__
#include <vector>
#include <glm/vec3.hpp>
#include <GL/glew.h>
#include <memory>


struct BoundingBox
{
	glm::vec3 minCoords = glm::vec3(std::numeric_limits<float>::infinity());
	glm::vec3 maxCoords = glm::vec3(-std::numeric_limits<float>::infinity());

	BoundingBox merge(const BoundingBox& other) const;
	BoundingBox merge(const glm::vec3& v) const;
	float getArea();
};

class Shape
{
public:
	virtual BoundingBox getBoundingBox() const = 0;
	virtual float approxSurfaceArea() const = 0;
	virtual glm::vec3 center() const = 0;
};

struct vec4 {
	float x, y, z, w;
};

struct Material {
	vec4 color;
	vec4 specularReflectivity;
	float refractiveIndex;
	int type; // 0 diffuse, 1 specular, 2 refractive

	int padding1;
	int padding2;
};

struct GPUFriendlyTriangle {
	vec4 a;
	vec4 b;
	vec4 c;
	Material material;
};

struct Triangle : public Shape {
	glm::vec3 a, b, c;
	Material material;

	BoundingBox getBoundingBox() const override;
	float approxSurfaceArea() const override;
	glm::vec3 center() const override;
	GPUFriendlyTriangle getGPUFriendly() const;
};

struct PointLightSource {
	vec4 position;
	vec4 intensity;  // vec3(r,g,b)
};


class SceneGeometry {
public:
	std::vector<GPUFriendlyTriangle> sceneGeometry;

	GPUFriendlyTriangle* getPointer()
	{
		// This is valid since by C++11 standard, vectors are guaranteed to be contiguous in memory
		return &sceneGeometry[0];
	}

	GLsizeiptr getSize()
	{
		return sizeof(GPUFriendlyTriangle) * sceneGeometry.size();
	}
};

class Lighting {
public:
	std::vector<PointLightSource> lightSources;

	PointLightSource* getPointer()
	{
		// This is valid since by C++11 standard, vectors are guaranteed to be contiguous in memory
		return &lightSources[0];
	}

	GLsizeiptr getSize()
	{
		return sizeof(PointLightSource) * lightSources.size();
	}
};

BoundingBox getBoundingBoxForAllShapes(const std::vector<Triangle>& shapes);

#endif