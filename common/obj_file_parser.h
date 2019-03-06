#include "../common/geometry.h"
#include "../common/bvh.h"
#include "../common/scene.h"
#include "../external/tiny_obj_loader.h"

// Builds a scene given an .obj file (and its corresponding .mtl and texture files).
Scene loadObj(const std::string& objPath, const std::string& mtlBaseDir, const std::string& textureBaseDir);