#pragma once

#include <vector>
#include "geometry.h"
#include "bvh.h"

// Represents an entire scene to be rendered by Photon. Refer to the GPUScene struct to see how this
// is represented on the GPU.
struct Scene
{
	// All spheres within the scene
	std::vector<Sphere> spheres;

	// All triangles stored as a BVH
	BVHGpuDataRaw triangleData;

	// All materials used by this scene
	std::vector<MaterialInfo> materials;

	// All light sources used by the scene
	std::vector<PointLightSource> lights;
};

// Specialization of the stream reader for Scene. Refer to streamWriter.h for more information.
template <>
Scene readFromStream<Scene>(std::istream& s);

// Specialization of the stream writer for Scene. Refer to streamWriter.h for more information.
template<>
void writeToStream<Scene>(std::ostream& s, const Scene& v);

