#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include "helper_math.h"

#define WIDTH 500
#define HEIGHT 500
#define NUM_PHOTONS (1<<20)

struct Photon{
    float3 pos;
    float3 power;
};

struct PointLight{
    float3 pos;
    float3 power;
};

struct Triangle{
    float3 v0,v1,v2;
    float3 color;
    __device__ float intersect(float3& point, float3& direction) const{
	float3 tvec = point - v0;
	float3 pvec = cross(direction, v2-v0);
	float det = dot(v1-v0, pvec);
        
	det = __fdividef(1.0, det);

	float u = dot(tvec, pvec)*det;
	if (u < 0 || u > 1)
	    return -1e20;

	float3 qvec = cross(tvec, v1-v0);

	float v = dot(direction, qvec) * det;

	if (v < 0 || (u+v) > 1)
	    return -1e20;

	return dot(v2-v0, qvec) * det;
    }
};


__constant__ PointLight pointLights[] = {
    {{0,0.95,0}, {1000000,1000000,1000000}}
};

__constant__ Triangle triangles[] = {
    {{-1,-1,-1},{1,-1,-1},{-1,-1,1},{0.9,0.9,0.9}},
    {{1,-1,-1},{1,-1,1},{-1,-1,1},{0.9,0.9,0.9}},
    {{-1,-1,-1},{-1,1,-1},{1,1,-1},{0.9,0.9,0.9}},
    {{-1,-1,-1},{1,1,-1},{1,-1,-1},{0.9,0.9,0.9}},
    {{-1,1,1},{1,1,1},{1,1,-1},{0.9,0.9,0.9}},
    {{-1,1,1},{1,1,-1},{-1,1,-1},{0.9,0.9,0.9}},
    {{-1,-1,1},{-1,1,1},{-1,1,-1},{0.9,0.2,0.2}},
    {{-1,-1,1},{-1,1,-1},{-1,-1,-1},{0.9,0.2,0.2}},
    {{1,-1,1},{1,1,1},{1,1,-1},{0.2,0.9,0.2}},
    {{1,-1,1},{1,1,-1},{1,-1,-1},{0.2,0.9,0.2}}
};

__global__ void getPhotonsKernel(Photon* photonList){
    uint idx = blockIdx.x*blockDim.x + threadIdx.x;

    curandState randState;
    curand_init(idx,0,0,&randState);

    float cosPhi = curand_uniform(&randState)*2 - 1;
    float sinPhi = sqrtf(1-cosPhi*cosPhi);
    float theta = curand_uniform(&randState)*2*M_PI;

    float3 origin = pointLights[0].pos;
    float3 direction = make_float3(sinPhi*cosf(theta), sinPhi*sinf(theta), cosPhi);

    float t = 1e20;
    uint triIdx = 0;
    uint numTriangles = sizeof(triangles) / sizeof(Triangle);
    for (uint i = numTriangles; i--; ){
	float d = triangles[i].intersect(origin, direction);
	if (d>0 && d<t){
	    t = d;
	    triIdx = i;
	}
    }
    if (t < 1e19){
	photonList[idx].pos = origin + direction*t;
	photonList[idx].power = triangles[triIdx].color*pointLights[0].power/NUM_PHOTONS;
    }
    else{
	photonList[idx].pos = make_float3(1e20,1e20,1e20);
    }
}

void writeTestToFile(std::vector<Photon> photons, std::string filename){
    std::vector<int> sums(3*WIDTH*HEIGHT);
    std::vector<uint> counts(WIDTH*HEIGHT);
    for (Photon photon : photons){
	if (photon.pos.x >= -1 && photon.pos.y >= -1 && photon.pos.z >= -1 &&
	    photon.pos.x <= 1 && photon.pos.y <= 1 && photon.pos.z <= 1){
	    int x = photon.pos.x/2/(-photon.pos.z + 2)*WIDTH + WIDTH/2;
	    int y = -photon.pos.y/2/(-photon.pos.z + 2)*HEIGHT + HEIGHT/2;
	    uint idx = y*WIDTH + x;
	    sums[3*idx+0] += (int)255*photon.power.x;
	    sums[3*idx+1] += (int)255*photon.power.y;
	    sums[3*idx+2] += (int)255*photon.power.z;
	    counts[idx] += 1;
	}
    }
    std::vector<char> pixels(3*WIDTH*HEIGHT);
    for (uint i = 0; i<counts.size(); ++i){
	if(counts[i]){
	    pixels[3*i+0] = sums[3*i+0]/counts[i];
	    pixels[3*i+1] = sums[3*i+1]/counts[i];
	    pixels[3*i+2] = sums[3*i+2]/counts[i];
	}
    }
    
    std::ofstream file;
    file.open(filename, std::ios::out | std::ios::binary);
    if (!file.is_open()){
	std::cerr << "Unable to save file" << std::endl;
	exit(1);
    }
    file << "P6\n" << WIDTH << " " << HEIGHT << "\n" << "255\n";
    
    file.write(pixels.data(), pixels.size()*sizeof(char));
    
    file.close();
}

int main(){

    std::vector<Photon> photonList_h(NUM_PHOTONS);
    Photon* photonList_d;
 
    cudaMalloc(&photonList_d, photonList_h.size()*sizeof(Photon));
    
    getPhotonsKernel<<<NUM_PHOTONS/64, 64>>>(photonList_d);
    cudaDeviceSynchronize();
    cudaMemcpy(photonList_h.data(), photonList_d,
	       photonList_h.size()*sizeof(Photon), cudaMemcpyDeviceToHost);
 
    cudaFree(photonList_d);
    
    writeTestToFile(photonList_h, "test.ppm");
        
    return 0;
}
