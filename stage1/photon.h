#ifndef __PHOTON_H
#define __PHOTON_H

//#include <cuda_runtime.h>

struct float3{
    float x,y,z;
};

struct Photon{
    float3 pos;
    float3 power;
};

#endif
