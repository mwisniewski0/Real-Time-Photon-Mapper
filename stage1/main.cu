#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include "helper_math.h"
#include "../common/photon.h"

#define WIDTH 500
#define HEIGHT 500
#define NUM_PHOTONS (1<<20)
#define MAX_DEPTH 10

enum ScatterType {DIFFUSE, SPECULAR, ABSORBED};

struct PointLight{
    float3 pos;
    float3 power;
};

struct Triangle{
    float3 v0,v1,v2; //verticies
    float3 diffuse; //diffuse reflectance
    float3 specular; //specular reflectance
    
    //uses moller trumbore algorithm
    //returns intersection distance on line or -infinity if line doesn't hit
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
    {{0,0.95,0}, {100000,100000,100000}}
};

//hard coded cornell box
__constant__ Triangle triangles[] = {
    //floor
    {{-1,-1,-1},{1,-1,-1},{-1,-1,1},{0.9,0.9,0.9},{0,0,0}},
    {{1,-1,-1},{1,-1,1},{-1,-1,1},{0.9,0.9,0.9},{0,0,0}},
    //back
    {{-1,-1,-1},{-1,1,-1},{1,1,-1},{0.9,0.9,0.9},{0,0,0}},
    {{-1,-1,-1},{1,1,-1},{1,-1,-1},{0.9,0.9,0.9},{0,0,0}},
    //ceiling
    {{-1,1,1},{1,1,1},{1,1,-1},{0.2,0.2,0.9},{0,0,0}},
    {{-1,1,1},{1,1,-1},{-1,1,-1},{0.2,0.2,0.9},{0,0,0}},
    //left
    {{-1,-1,1},{-1,1,1},{-1,1,-1},{0.0,0.0,0.0},{0.9,0.2,0.2}},
    {{-1,-1,1},{-1,1,-1},{-1,-1,-1},{0.0,0.0,0.0},{0.9,0.2,0.2}},
    //right
    {{1,-1,1},{1,1,-1},{1,1,1},{0.0,0.0,0.0},{0.2,0.9,0.2}},
    {{1,-1,1},{1,-1,-1},{1,1,-1},{0.0,0.0,0.0},{0.2,0.9,0.2}}
};

//trace photons and return array in photonList
__global__ void getPhotonsKernel(Photon* photonList){
    uint idx = blockIdx.x*blockDim.x + threadIdx.x; //gpu thread index

    curandState randState;
    curand_init(idx,0,10,&randState); //the 10 is an offset which seems to fix banding but more investigation is needed

    float cosPhi = curand_uniform(&randState)*2 - 1;
    float sinPhi = sqrtf(1-cosPhi*cosPhi);
    float theta = curand_uniform(&randState)*2*M_PI;

    float3 origin = pointLights[0].pos;
    float3 direction = make_float3(sinPhi*cosf(theta), sinPhi*sinf(theta), cosPhi);
    float3 color = make_float3(1,1,1);
    
    uint depth = MAX_DEPTH;

    int count = 0;
    for (; depth--; ){
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

	float3 diffuse = triangles[triIdx].diffuse;
	float3 specular = triangles[triIdx].specular;
	float d_avg = (diffuse.x + diffuse.y + diffuse.z)/3;
	float s_avg = (specular.x + specular.y + specular.z)/3;
	float xi = curand_uniform(&randState);
	ScatterType action;
	if(xi < d_avg){
	    action = DIFFUSE;
	}
	else if (xi < d_avg + s_avg){
	    action = SPECULAR;
	}
	else{
	    action = ABSORBED;
	}

	if (t < 1e19){
	    origin+=direction*t;
	    if ((action == DIFFUSE || action == ABSORBED)){
		photonList[MAX_DEPTH*idx + count].pos = origin;
		photonList[MAX_DEPTH*idx + count].power = color*pointLights[0].power/NUM_PHOTONS;
		++count;
	    }
	}
	else{ // hit nothing
	    break;
	}

	float3 normal = cross(triangles[triIdx].v2 - triangles[triIdx].v0,
			      triangles[triIdx].v1 - triangles[triIdx].v0);
	normal = normalize(normal);
	

	if (action == DIFFUSE){
	    cosPhi = curand_uniform(&randState);
	    sinPhi = sqrtf(1-cosPhi*cosPhi);
	    theta = curand_uniform(&randState)*2*M_PI;
	    
	    float3 w = normal;
	    float3 u = normalize(cross((fabs(w.x) > 0.0001 ?
					make_float3(0,1,0) :
					make_float3(1,0,0)), w));
	    float3 v = cross(w,u);
	    direction = normalize(u*cosf(theta)*sinPhi +
				  v*sinf(theta)*sinPhi +
				  w*cosPhi);
	    color*=diffuse;
	}
	else if(action == SPECULAR){
	    direction = direction - 2*normal*dot(normal, direction);
	    color*=specular;
	}
	else{//absorbed	    
	    break;
	}
    }
    for(uint i = 0; i<count; ++i){
	photonList[MAX_DEPTH*idx + i].power/=count;
    }

}

void writeTestToFile(std::vector<Photon> photons, std::string filename){
    std::vector<int> sums(3*WIDTH*HEIGHT);
    for (Photon photon : photons){
	if (photon.pos.x >= -1 && photon.pos.y >= -1 && photon.pos.z >= -1 &&
	    photon.pos.x <= 1 && photon.pos.y <= 1 && photon.pos.z <= 1){
	    int x = photon.pos.x/2/(-photon.pos.z + 2)*WIDTH + WIDTH/2;
	    int y = -photon.pos.y/2/(-photon.pos.z + 2)*HEIGHT + HEIGHT/2;
	    uint idx = y*WIDTH + x;
	    float theta = atan(sqrt(photon.pos.x*photon.pos.x + photon.pos.y*photon.pos.y)/(photon.pos.z+2));
	    sums[3*idx+0] += (int)255*photon.power.x;
	    sums[3*idx+1] += (int)255*photon.power.y;
	    sums[3*idx+2] += (int)255*photon.power.z;
	}
    }
    std::vector<char> pixels(3*WIDTH*HEIGHT);
    for (uint i = 0; i<WIDTH*HEIGHT; ++i){
	pixels[3*i+0] = clamp(sums[3*i+0],0,255);
	pixels[3*i+1] = clamp(sums[3*i+1],0,255);
	pixels[3*i+2] = clamp(sums[3*i+2],0,255);
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

    std::vector<Photon> photonList_h(NUM_PHOTONS*MAX_DEPTH);
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
