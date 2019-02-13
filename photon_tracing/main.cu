#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include "../common/cutil_math.h"
#include "../common/geometry.h"
#include "../common/photon.h"

#include "../cuda_common/cudaHelpers.h"
#include "../cuda_common/gpuBvh.h"
#include "../cuda_common/gpuScene.h"
#include "../cuda_common/kd-tree.h"

#define WIDTH 500
#define HEIGHT 500
#define NUM_PHOTONS (1<<20)
#define MAX_DEPTH 10

enum ScatterType {DIFFUSE, SPECULAR, ABSORBED};

/*
 * Triangle2 and createScene are very temporary in order to hgave a simple scene to do photon tracing.
 * Delete these and use obj_parser when it is done
 */
struct Triangle2
{
	float3 a, b, c;
	Material material;

	Triangle toTriangle()
	{
		return Triangle::from3Points(a, b, c, material);
	}
};

SceneInfo createScene(){
    	std::vector<Triangle> triangles;// = loadTriangles("models/bun_zipper.ply", Material{ {0,1,1}, {0.8f, 0.8f, 0.8f}, 2.5f, 0x0000 });

	Scene scene;

	Triangle2 t;

	// Back wall
	t.a = { -1, 1, 1 };
	t.b = { 1, 1, 1 };
	t.c = { -1, -1, 1 };
	t.material = {
		{ 1, 1, 1 }, // color
		{ 0, 0, 0 }, // reflectivity
		2.5,  // refractive index (diamond)
		0x0000 // type
	};
	triangles.push_back(t.toTriangle());
	t.a = { 1, 1, 1 };
	t.b = { 1, -1, 1 };
	t.c = { -1, -1, 1 };
	t.material = {
		{ 1, 1, 1 }, // color
		{ 0, 0, 0 }, // reflectivity
		2.5,  // refractive index (diamond)
		0x0000 // type
	};
	triangles.push_back(t.toTriangle());
	
	// Front wall
	t.a = { -1, 1, -4 };
	t.b = { 1, 1, -4 };
	t.c = { -1, -1, -4 };
	t.material = {
		{ 0, 1, 0 }, // color
		{ 0.0f, 0.0f, 0.0f }, // reflectivity
		2.5,  // refractive index (diamond)
		0x0000 // type
	};
	triangles.push_back(t.toTriangle());

	t.a = { 1, 1, -4 };
	t.b = { 1, -1, -4 };
	t.c = { -1, -1, -4 };
	t.material = {
		{ 0, 1, 0 }, // color
		{ 0.0f, 0.0f, 0.0f }, // reflectivity
		2.5,  // refractive index (diamond)
		0x0000 // type
	};
	triangles.push_back(t.toTriangle());

	// Left wall
	t.a = { -1, -1, 1 };
	t.b = { -1, 1, 1 };
	t.c = { -1, -1, -4 };
	t.material = {
		{ 0, 0, 1 }, // color
		{ 0.0f, 0.0f, 0.0f }, // reflectivity
		2.5,  // refractive index (diamond)
		0x0000 // type
	};
	triangles.push_back(t.toTriangle());

	t.a = { -1, 1, 1 };
	t.b = { -1, 1, -4 };
	t.c = { -1, -1, -4 };
	t.material = {
		{ 0, 0, 1 }, // color
		{ 0.0f, 0.0f, 0.0f }, // reflectivity
		2.5,  // refractive index (diamond)
		0x0000 // type
	};
	triangles.push_back(t.toTriangle());

	// Right wall
	t.a = { 1, -1, 1 };
	t.b = { 1, 1, 1 };
	t.c = { 1, -1, -4 };
	t.material = {
		{ 1, 1, 0 }, // color
		{ 0, 0.0f, 0.0f }, // reflectivity
		2.5,  // refractive index (diamond)
		0x0000 // type
	};
	triangles.push_back(t.toTriangle());
	t.a = { 1, 1, 1 };
	t.b = { 1, 1, -4 };
	t.c = { 1, -1, -4 };
	t.material = {
		{ 1, 1, 0 }, // color
		{ 0.f, 0.0f, 0.0f }, // reflectivity
		2.5,  // refractive index (diamond)
		0x0000 // type
	};
	triangles.push_back(t.toTriangle());


	// Top wall
	t.a = { -1, 1, 1 };
	t.b = { 1, 1, 1 };
	t.c = { -1, 1, -4 };
	t.material = {
		{ 0, 1, 1 }, // color
		{ 0.0f, 0.0f, 0.0f }, // reflectivity
		2.5,  // refractive index (diamond)
		0x0000 // type
	};
	triangles.push_back(t.toTriangle());


	t.a = { 1, 1, 1 };
	t.b = { 1, 1, -4 };
	t.c = { -1, 1, -4 };
	t.material = {
		{ 0, 1, 1 }, // color
		{ 0.0f, 0.0f, 0.0f }, // reflectivity
		2.5,  // refractive index (diamond)
		0x0000 // type
	};
	triangles.push_back(t.toTriangle());


	// Bottom wall
	t.a = { -1, -1, 1 };
	t.b = { 1, -1, 1 };
	t.c = { -1, -1, -4 };
	t.material = {
		{ 1, 0, 1 }, // color
		{ 0.0f, 0.0f, 0.0f }, // reflectivity
		2.5,  // refractive index (diamond)
		0x0000 // type
	};
	triangles.push_back(t.toTriangle());


	t.a = { 1, -1, 1 };
	t.b = { 1, -1, -4 };
	t.c = { -1, -1, -4 };
	t.material = {
		{ 1, 0, 1 }, // color
		{ 0.0f, 0.0f, 0.0f }, // reflectivity
		2.5,  // refractive index (diamond)
		0x0000 // type
	};
	triangles.push_back(t.toTriangle());
	
	scene.triangles = std::move(triangles);

	scene.lights.push_back(PointLightSource{
		{0, 0.0f, 0},
		{1000000, 1000000, 1000000},
		});
	return SceneInfo::fromScene(scene);
}

/*
 * CUDA kernel to trace photons. Geometry is defined by scene and output is stored in photonList.
 */
__global__ void getPhotonsKernel(SceneInfo scene, Photon* photonList) {
	uint idx = blockIdx.x*blockDim.x + threadIdx.x; //gpu thread index

	curandState randState;
	curand_init(idx, 0, 10, &randState); //the 10 is an offset which seems to fix banding but more investigation is needed

	float cosPhi = curand_uniform(&randState) * 2 - 1;
	float sinPhi = sqrtf(1 - cosPhi*cosPhi);
	float theta = curand_uniform(&randState) * 2 * M_PI;


	float3 origin = scene.lights[0].position;
	float3 direction = make_float3(sinPhi*cosf(theta), sinPhi*sinf(theta), cosPhi);
	Ray ray = { origin, direction };
	float3 color = scene.lights[0].intensity;


	int count = 0;
	for (uint depth = MAX_DEPTH; depth > 0; --depth) {
		float t = 1e20;
		Triangle* tri = scene.triangleBvh.intersectRay(ray, t);
		if (tri) {
			Material& mat = tri->material;
			float3 diffuse = mat.color;
			float3 specular = mat.specularReflectivity;
			float d_avg = (diffuse.x + diffuse.y + diffuse.z) / 3;
			float s_avg = (specular.x + specular.y + specular.z) / 3;
			float xi = curand_uniform(&randState);
			ScatterType action;
			if (xi < d_avg) {
				action = DIFFUSE;
			}
			else if (xi < d_avg + s_avg) {
				action = SPECULAR;
			}
			else {
				action = ABSORBED;
			}

			if (t < 1e19) {
				ray.origin += direction*t;
				if ((action == DIFFUSE || action == ABSORBED)) {
					photonList[MAX_DEPTH*idx + count].pos = ray.origin;
					photonList[MAX_DEPTH*idx + count].power = color / NUM_PHOTONS;
					auto p = photonList[MAX_DEPTH*idx + count].power;
					printf("%f,%f,%f\n", p.x, p.y, p.z);
					++count;
				}
			}
			else { // hit nothing
				break;
			}

			float3 normal = tri->normal;
			normal = normalize(normal);


			if (action == DIFFUSE) {
				cosPhi = curand_uniform(&randState);
				sinPhi = sqrtf(1 - cosPhi*cosPhi);
				theta = curand_uniform(&randState) * 2 * M_PI;

				float3 w = normal;
				float3 u = normalize(cross((fabs(w.x) > 0.0001 ?
					make_float3(0, 1, 0) :
					make_float3(1, 0, 0)), w));
				float3 v = cross(w, u);
				direction = normalize(u*cosf(theta)*sinPhi +
					v*sinf(theta)*sinPhi +
					w*cosPhi);
				color *= diffuse;
			}
			else if (action == SPECULAR) {
				ray.dir = direction - 2 * normal*dot(normal, direction);
				color *= specular;
			}
			else {//absorbed	    
				break;
			}
		}
		else {
			break;
		}
	}

	for (uint i = 0; i < count; ++i) {
		photonList[MAX_DEPTH*idx + i].power /= count;
	}

}

/*
 * Simple functon to get some idea of if photon tracing is working correctly. Simply projects all
 * photons in photons to the screen. Also clips to [-1,1]^3. Saves a .ppm image to filename.
 */
void writeTestToFile(std::vector<Photon> photons, std::string filename) {
	std::vector<int> sums(3 * WIDTH*HEIGHT);
	for (Photon photon : photons) {
		if (photon.pos.x >= -1 && photon.pos.y >= -1 && photon.pos.z >= -1 &&
			photon.pos.x <= 1 && photon.pos.y <= 1 && photon.pos.z <= 1) {
			int x = photon.pos.x / 2 / (-photon.pos.z + 2)*WIDTH + WIDTH / 2;
			int y = -photon.pos.y / 2 / (-photon.pos.z + 2)*HEIGHT + HEIGHT / 2;
			uint idx = y*WIDTH + x;
			float theta = atan(sqrt(photon.pos.x*photon.pos.x + photon.pos.y*photon.pos.y) / (photon.pos.z + 2));
			sums[3 * idx + 0] += (int)255 * photon.power.x;
			sums[3 * idx + 1] += (int)255 * photon.power.y;
			sums[3 * idx + 2] += (int)255 * photon.power.z;
		}
	}
	std::vector<char> pixels(3 * WIDTH*HEIGHT);
	for (uint i = 0; i < WIDTH*HEIGHT; ++i) {
		pixels[3 * i + 0] = clamp(sums[3 * i + 0], 0, 255);
		pixels[3 * i + 1] = clamp(sums[3 * i + 1], 0, 255);
		pixels[3 * i + 2] = clamp(sums[3 * i + 2], 0, 255);
	}

	std::ofstream file;
	file.open(filename, std::ios::out | std::ios::binary);
	if (!file.is_open()) {
		std::cerr << "Unable to save file" << std::endl;
		exit(1);
	}
	file << "P6\n" << WIDTH << " " << HEIGHT << "\n" << "255\n";

	file.write(pixels.data(), pixels.size() * sizeof(char));

	file.close();
}

int main(){

    std::vector<Photon> photonList_h(NUM_PHOTONS*MAX_DEPTH);
    Photon* photonList_d;
    
    checkCudaError(cudaMalloc(&photonList_d, photonList_h.size()*sizeof(Photon)));

    SceneInfo scene = createScene();
    
	getPhotonsKernel<<<NUM_PHOTONS/64, 64>>>(scene, photonList_d);
	checkCudaError(cudaGetLastError());
    cudaDeviceSynchronize();
    cudaMemcpy(photonList_h.data(), photonList_d,
	       photonList_h.size()*sizeof(Photon), cudaMemcpyDeviceToHost);
 
    cudaFree(photonList_d);
    
    writeTestToFile(photonList_h, "test.ppm");

    //sortPhotons(photonList_h);

#ifdef WIN32
	std::cin.ignore();
#endif
        
    return 0;
}
