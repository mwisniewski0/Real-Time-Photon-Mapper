#include "../common/geometry.h"
#include "../common/bvh.h"
#include "../common/scene.h"
#include "../external/tiny_obj_loader.h"

Scene loadObj(const std::string& objPath, const std::string& mtlBaseDir, const std::string& textureBaseDir);