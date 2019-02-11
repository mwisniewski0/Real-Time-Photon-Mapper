#pragma once

#include <vector>
#include "geometry.h"

struct Scene
{
	std::vector<Sphere> spheres;
	std::vector<Triangle> triangles;
	std::vector<PointLightSource> lights;
};
