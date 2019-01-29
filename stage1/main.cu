#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "cutil_math.h"

#define WIDTH 500
#define HEIGHT 500
#define NUM_PHOTONS WIDTH*HEIGHT

struct Photon{
    float3 pos;
    float3 power;
};

__global__ void getPhotonsKernel(Photon* photonList){
    uint x = blockIdx.x*blockDim.x + threadIdx.x;
    uint y = blockIdx.y*blockDim.y + threadIdx.y;

    uint idx = y*WIDTH + x;
    photonList[idx].pos = make_float3(2*(float)x/WIDTH - 1,1, (float)2*y/HEIGHT-1);
    photonList[idx].power = make_float3((float)x/WIDTH, (float)y/HEIGHT,0);
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
	    sums[3*idx+0] += (int)256*photon.power.x;
	    sums[3*idx+1] += (int)256*photon.power.y;
	    sums[3*idx+2] += (int)256*photon.power.z;
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
    dim3 block(8,8,1);
    dim3 grid(WIDTH/block.x, HEIGHT/block.y, 1);
    
    getPhotonsKernel<<<grid, block>>>(photonList_d);
    cudaDeviceSynchronize();
    cudaMemcpy(photonList_h.data(), photonList_d,
	       photonList_h.size()*sizeof(Photon), cudaMemcpyDeviceToHost);
 
    cudaFree(photonList_d);
    
    writeTestToFile(photonList_h, "test.ppm");
        
    return 0;
}
