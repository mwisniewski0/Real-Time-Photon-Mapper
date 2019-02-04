#ifndef __PLY_H_INCLUDED
#define __PLY_H_INCLUDED
#include <vector>
#include "Geometry.h"
#include <string>


std::vector<Triangle> loadTriangles(const std::string& path, Material m);

#endif
