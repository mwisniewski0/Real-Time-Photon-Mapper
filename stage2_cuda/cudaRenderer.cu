#ifdef _WIN64
#include <windows.h>
#endif
#include "cuda.h"
#include "cudaRenderer.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_gl_interop.h"
#include "../common/geometry.h"
#include "../common/cutil_math.h"
#include <iostream>
#include "../cuda_common/gpuBvh.h"
#include "../cuda_common/gpuScene.h"
#include "../stage1/photonTracing.h"
#include "../cuda_common/kd-tree.h"

__device__ const float EPSILON = 0.00001;
__device__ const float AIR_REFRACTIVE_INDEX = 1;

__device__ unsigned xpix = 0;
__device__ unsigned ypix = 0;


__device__ Ray getCameraRay(const CudaCamera& cam, float screenX, float screenY) {
	float3 pointOnScreen = (cam.screenBottomRight - cam.screenBottomLeft) * screenX +
		(cam.screenTopLeft - cam.screenBottomLeft) * screenY + cam.screenBottomLeft;
	return Ray::fromPoints(cam.eyePos, pointOnScreen);
}

__device__ float3 getNormalAwayFromRay(Ray ray, Triangle t) {
	// Avoiding branching - this will flip the sign of the normal if the normal forms an obtuse
	// angle with the direction of the ray
	return normalize(t.normal*dot(t.normal, -ray.dir));
}

struct RayHit {
	float3 pointOfHit;
	float3 normal;  // Unit vector
	Material material;
};

__device__ bool intersectRayAndSphere(const Ray& ray, const Sphere& s, RayHit* hit)
{
	float3 co = ray.origin - s.center;
	float b = 2 * dot(co, ray.dir);
	float c = dot(co, co) - s.radius*s.radius;
	float delta = b*b - 4 * c; // Since a is 1 (unitdir dot unitdir)

	if (delta < 0) return false;

	float sqrtDelta = sqrt(delta);
	float negT = (-b - sqrtDelta) / 2;
	float posT = (-b + sqrtDelta) / 2;

	if (negT <= 0 && posT <= 0)
	{
		// The sphere is behind the ray origin
		return false;
	}

	float dFromRayStart;
	bool collidedInside = false;
	if (negT <= 0 && posT > 0)
	{
		// We hit the sphere from the inside
		dFromRayStart = posT;
		collidedInside = true;
	}
	else
	{
		// Take the closer point of intersection
		dFromRayStart = negT;
	}
	hit->pointOfHit = ray.pointAtDistance(dFromRayStart);
	hit->normal = (hit->pointOfHit - s.center) * (1.0 / s.radius);
	hit->material = s.material;
	return true;
}

__device__ float lengthSquared(float3 v)
{
	return dot(v, v);
}

__device__ bool castRay(const Ray& ray, const SceneInfo& scene, RayHit*  result) {
	float closestHitDistance = 1e20;  // Infinity

	RayHit tempResult;
	auto intersectedTriangle = scene.triangleBvh.intersectRay(ray, closestHitDistance);
	if (intersectedTriangle != nullptr)
	{
		result->pointOfHit = ray.pointAtDistance(closestHitDistance);
		result->normal = getNormalAwayFromRay(ray, *intersectedTriangle);
		result->material = intersectedTriangle->material;
	}

	for (int i = 0; i < scene.spheres.size; ++i)
	{
		if (intersectRayAndSphere(ray, scene.spheres.contents[i], &tempResult))
		{
			float distanceSquared = length(ray.origin - tempResult.pointOfHit);
			if (closestHitDistance > distanceSquared)
			{
				closestHitDistance = distanceSquared;
				*result = tempResult;
			}
		}
	}

	return closestHitDistance < 1e19;
}

__device__ bool castRayNaive(const Ray& ray, const SceneInfo& scene, RayHit*  result) {
	float closestHitDistance = 1e20;  // Infinity
	//printf("%f %f %f, %f %f %f\n", ray.origin.x, ray.origin.y, ray.origin.z, ray.dir.x, ray.dir.y, ray.dir.z);

	RayHit tempResult;
	for (int i = 0; i < scene.triangleBvh.triangles.size; ++i)
	{
		float d = scene.triangleBvh.triangles.contents[i].intersect(ray);
		if (d > 0)
		{
			if (closestHitDistance > d)
			{
				closestHitDistance = d;
				result->pointOfHit = ray.pointAtDistance(closestHitDistance);
				result->normal = getNormalAwayFromRay(ray, scene.triangleBvh.triangles.contents[i]);
				result->material = scene.triangleBvh.triangles.contents[i].material;
			}
		}
	}

	for (int i = 0; i < scene.spheres.size; ++i)
	{
		if (intersectRayAndSphere(ray, scene.spheres.contents[i], &tempResult))
		{
			float distanceSquared = length(ray.origin - tempResult.pointOfHit);
			if (closestHitDistance > distanceSquared)
			{
				closestHitDistance = distanceSquared;
				*result = tempResult;
			}
		}
	}

	return closestHitDistance < 1e19;
}

__device__ float3 getHitIllumination(const SceneInfo& scene, const GPUVector<Photon>& photonMap, const RayHit& hit) {	
	float3 illumination = gatherPhotons(hit.pointOfHit, photonMap);
	//printf("%f %f %f\n", illumination.x, illumination.y, illumination.z);
	return illumination* hit.material.color;
	// return gatherPhotons(hit.pointOfHit, photonMap) * hit.material.color;
	// float3 illumination = make_float3(0.2, 0.2, 0.2) * hit.material.color;
	// for (int i = 0; i < scene.lights.size; ++i)
	// {
	// 	float3 vectorToLight = scene.lights.contents[i].position - hit.pointOfHit;
	//
	// 	RayHit hitTowardsLight;
	// 	bool lightReached = !castRay(
	// 		Ray::fromPoints(hit.pointOfHit + (hit.normal * EPSILON), scene.lights.contents[i].position),
	// 		scene, &hitTowardsLight);
	// 	lightReached =
	// 		lightReached || (
	// 			lengthSquared(hitTowardsLight.pointOfHit - hit.pointOfHit) >
	// 			lengthSquared(vectorToLight));
	// 	if (lightReached)
	// 	{
	// 		illumination += hit.material.color * scene.lights.contents[i].intensity * dot(normalize(vectorToLight), hit.normal);
	// 	}
	// }
	//
	//
	// return illumination;
}

__device__ float3 refract(const float3& in, const float3& normal, float eta)
{
	float nDotI = dot(in, normal);
	float k = 1.f - eta * eta * (1.f - nDotI * nDotI);
	if (k < 0.f)
		return make_float3(0.f, 0.f, 0.f);
	else
		return eta * in - (eta * nDotI + sqrtf(k)) * normal;
}

struct RaySplit
{
	Ray ray;
	float3 currentModifier;
};

__device__ float calculateReflectRatio(float n1, float n2, const float3& normal, const float3& incident)
{
	// Using Schlick's approximation

	float r0 = (n1 - n2) / (n1 + n2);
	r0 *= r0;
	float cosX = -dot(normal, incident);
	if (n1 > n2)
	{
		float n = n1 / n2;
		float sinT2 = n*n*(1.0 - cosX*cosX);
		// Total internal reflection
		if (sinT2 > 1.0)
			return 1.0;
		cosX = sqrt(1.0 - sinT2);
	}
	float x = 1.0 - cosX;
	float ret = r0 + (1.0 - r0)*(x*x*x*x*x);

	return ret;
}

__device__ float3 getRayColor(const SceneInfo& scene, const GPUVector<Photon>& photonMap, Ray ray) {
	const int MAX_RAY_BOUNCE = 6;

	RaySplit rays[MAX_RAY_BOUNCE];
	rays[0] = { ray, 1.0f, 1.0f, 1.0f };
	int lastRay = 0;

	float3 currentIllumination = make_float3(0);

	for (int bounce = 0; bounce < MAX_RAY_BOUNCE; ++bounce)
	{
		if (bounce > lastRay) break;

		const auto& split = rays[bounce];
		const auto& r = split.ray;

		RayHit hit;
		if (!castRay(r, scene, &hit))
		{
			break;
		}

		if (hit.material.type == 1)
		{
			// Reflective
			++lastRay;
			if (lastRay == MAX_RAY_BOUNCE) continue;
			rays[lastRay].currentModifier = split.currentModifier * hit.material.specularReflectivity;
			rays[lastRay].ray.origin = hit.pointOfHit + (hit.normal * EPSILON);
			rays[lastRay].ray.dir = reflect(r.dir, hit.normal);
		}
		else if (hit.material.type == 2)
		{
			// Refractive
			if (dot(hit.normal, r.dir) < 0)
			{
				float reflectRatio = calculateReflectRatio(AIR_REFRACTIVE_INDEX, hit.material.refractiveIndex, hit.normal, r.dir);

				++lastRay;
				if (lastRay == MAX_RAY_BOUNCE) continue;
				rays[lastRay].currentModifier = split.currentModifier * hit.material.specularReflectivity * (1.0f - reflectRatio);
				rays[lastRay].ray.origin = hit.pointOfHit + (-hit.normal * EPSILON);
				rays[lastRay].ray.dir = refract(r.dir, hit.normal, 1.0 / hit.material.refractiveIndex);

				++lastRay;
				if (lastRay == MAX_RAY_BOUNCE) continue;
				rays[lastRay].currentModifier = split.currentModifier * hit.material.specularReflectivity * reflectRatio;
				rays[lastRay].ray.origin = hit.pointOfHit + (hit.normal * EPSILON);
				rays[lastRay].ray.dir = reflect(r.dir, hit.normal);
			}
			else
			{
				float reflectRatio = calculateReflectRatio(hit.material.refractiveIndex, AIR_REFRACTIVE_INDEX, -hit.normal, r.dir);

				++lastRay;
				if (lastRay == MAX_RAY_BOUNCE) continue;
				rays[lastRay].currentModifier = split.currentModifier * hit.material.specularReflectivity * (1.0f - reflectRatio);
				rays[lastRay].ray.origin = hit.pointOfHit + (hit.normal * EPSILON);
				rays[lastRay].ray.dir = refract(r.dir, -hit.normal, 1.0 / hit.material.refractiveIndex);

				++lastRay;
				if (lastRay == MAX_RAY_BOUNCE) continue;
				rays[lastRay].currentModifier = split.currentModifier * hit.material.specularReflectivity * reflectRatio;
				rays[lastRay].ray.origin = hit.pointOfHit + (hit.normal * EPSILON);
				rays[lastRay].ray.dir = reflect(r.dir, hit.normal);
			}
		}
		else
		{
			// Diffuse
			currentIllumination += split.currentModifier * getHitIllumination(scene, photonMap, hit);
		}
	}

	return currentIllumination;
}


__global__ void renderingKernel(CudaCamera cam, GPUVector<Photon> photonMap, SceneInfo scene,
                                unsigned screenWidth, unsigned screenHeight, cudaSurfaceObject_t output)
{
	// Calculate surface coordinates
	unsigned int xPixel = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int yPixel = blockIdx.y * blockDim.y + threadIdx.y;

	xpix = xPixel;
	ypix = yPixel;

	Ray ray = getCameraRay(cam, xPixel / (float)screenWidth, yPixel / (float)screenHeight);

	uchar4 color;
	float3 colorFloat = getRayColor(scene, photonMap, ray);

	colorFloat = minf3(colorFloat, make_float3(1.0));

	color.x = (unsigned char)(colorFloat.x * 255);
	color.y = (unsigned char)(colorFloat.y * 255);
	color.z = (unsigned char)(colorFloat.z * 255);

	// if (xPixel == 0 && yPixel == 0)
	// {
	// 	printf("%f %f %f\n", scene.spheres[0].center.x, scene.spheres[0].center.y, scene.spheres[0].center.z);
	// 	printf("%d %d %d\n", scene.trianglesCount, scene.spheresCount, scene.lightsCount);
	// }

	surf2Dwrite(color, output, xPixel * 4, yPixel);
}

CudaRenderer::CudaRenderer(int outputWidth, int outputHeight)
{
	this->outputWidth = outputWidth;
	this->outputHeight = outputHeight;

	initialize();
}

template <typename T>
T* loadVectorToGpu(const std::vector<T> v)
{
	T* device_mem;
	auto err = cudaMalloc((void**)(&device_mem), v.size() * sizeof(T));
	err = cudaMemcpy((void*)(device_mem), (void*)(&(v[0])),
	                 v.size() * sizeof(T), cudaMemcpyHostToDevice);
	return device_mem;
}

void CudaRenderer::loadScene(const Scene& scene)
{
	this->scene = SceneInfo::fromScene(scene);
	this->photonMap = vectorToGpu(tracePhotons(this->scene, 1 << 17));
}

void CudaRenderer::renderFrame(const Camera& camera)
{
	cudaGraphicsMapResources(1, &viewCudaResource);
	{
		cudaArray_t viewCudaArray;
		cudaGraphicsSubResourceGetMappedArray(&viewCudaArray, viewCudaResource, 0, 0);
		cudaResourceDesc viewCudaArrayResourceDesc;
		{
			viewCudaArrayResourceDesc.resType = cudaResourceTypeArray;
			viewCudaArrayResourceDesc.res.array.array = viewCudaArray;
		}
		cudaSurfaceObject_t viewCudaSurfaceObject;
		cudaCreateSurfaceObject(&viewCudaSurfaceObject, &viewCudaArrayResourceDesc);
		{
			dim3 block(8, 8, 1);
			dim3 grid(outputWidth / block.x, outputHeight / block.y, 1);
			renderingKernel << <grid, block >> >(camera.getCudaInfo(), photonMap, scene, outputWidth, outputHeight, viewCudaSurfaceObject);
			checkCudaError(cudaGetLastError());
		}
		cudaDestroySurfaceObject(viewCudaSurfaceObject);
	}
	cudaGraphicsUnmapResources(1, &viewCudaResource);

	cudaStreamSynchronize(0);

	glBindTexture(GL_TEXTURE_2D, viewGLTexture);
	{
		glBegin(GL_QUADS);
		{
			glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f, -1.0f);
			glTexCoord2f(1.0f, 0.0f); glVertex2f(+1.0f, -1.0f);
			glTexCoord2f(1.0f, 1.0f); glVertex2f(+1.0f, +1.0f);
			glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f, +1.0f);
		}
		glEnd();
	}
	glBindTexture(GL_TEXTURE_2D, 0);

	glFinish();
}

void CudaRenderer::initialize()
{
	unsigned deviceCount;
	auto const DEVICE_LIST_BUFFER_SIZE = 1;
	int cudaDevices[DEVICE_LIST_BUFFER_SIZE];
	cudaGLGetDevices(&deviceCount, cudaDevices, DEVICE_LIST_BUFFER_SIZE, cudaGLDeviceListCurrentFrame);
	cudaSetDevice(cudaDevices[0]);

	glEnable(GL_TEXTURE_2D);
	glGenTextures(1, &viewGLTexture);

	glBindTexture(GL_TEXTURE_2D, viewGLTexture);
	{
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, outputWidth, outputHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	}
	glBindTexture(GL_TEXTURE_2D, 0);

	cudaGraphicsGLRegisterImage(&viewCudaResource, viewGLTexture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
}
