#ifndef __PLY_H_INCLUDED
#define __PLY_H_INCLUDED
#include <vector>
#include "geometry.h"
#include <string>

// Loads the triangles of the .ply file residing at the provided path. Sets all the triangles to use
// the specified material index.
std::vector<Triangle> loadTriangles(const std::string& path, unsigned materialIndex);

#endif
