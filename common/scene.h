#pragma once

#include <vector>
#include "geometry.h"
#include "bvh.h"

struct Scene
{
	std::vector<Sphere> spheres;
	BVHGpuDataRaw triangleData;
	std::vector<MaterialInfo> materials;
	std::vector<PointLightSource> lights;
};

template <>
Scene readFromStream<Scene>(std::istream& s);

template<>
void writeToStream<Scene>(std::ostream& s, const Scene& v);

