#ifndef __PHOTON_H
#define __PHOTON_H

#include "../common/cutil_math.h"

// Represents a single photon when running photon-tracing.
struct Photon{
    float3 pos;
	float3 power;
};

#endif
