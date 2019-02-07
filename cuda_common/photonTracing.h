#pragma once
#include "../common/photon.h"
#include <vector>
#include "../cuda_common/gpuScene.h"

std::vector<Photon> tracePhotons(const SceneInfo& scene, int numPhotons);
