#ifndef GEOMETRY_H_INCLUDED__
#define GEOMETRY_H_INCLUDED__
#include <vector>

// TODO: These structures are used to transfer 

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

struct Triangle {
	vec4 a;
	vec4 b;
	vec4 c;
	Material material;
};

struct PointLightSource {
	vec4 position;
	vec4 intensity;  // vec3(r,g,b)
};


class SceneGeometry {
public:
	std::vector<Triangle> sceneGeometry;

	Triangle* getPointer()
	{
		// This is valid since by C++11 standard, vectors are guaranteed to be contiguous in memory
		return &sceneGeometry[0];
	}

	GLsizeiptr getSize()
	{
		return sizeof(Triangle) * sceneGeometry.size();
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

#endif