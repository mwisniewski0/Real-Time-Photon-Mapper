#ifdef _WIN64
#include <windows.h>
#endif

#include "cudaRenderer.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_gl_interop.h"
#include "Geometry.h"
#include "helper_math.h"
#include <iostream>

__device__ const float EPSILON = 0.00001;
__device__ const float AIR_REFRACTIVE_INDEX = 1;

__device__ unsigned xpix = 0;
__device__ unsigned ypix = 0;



struct Ray {
	float3 origin;
	float3 dir;  // Unit dir vector

	__device__ static Ray fromPoints(const float3& start, const float3& through)
	{
		Ray ray;
		ray.origin = start;
		ray.dir = normalize(through - start);
		return ray;
	}

	__device__ float3 pointAtDistance(float distance) const
	{
		return origin + (dir * distance);
	}
};

__device__ Ray getCameraRay(const CudaCamera& cam, float screenX, float screenY) {
	float3 pointOnScreen = (cam.screenBottomRight - cam.screenBottomLeft) * screenX +
		(cam.screenTopLeft - cam.screenBottomLeft) * screenY + cam.screenBottomLeft;
	return Ray::fromPoints(cam.eyePos, pointOnScreen);
}

struct RayHit {
	float3 pointOfHit;
	float3 normal;  // Unit vector
	Material material;
};

__device__ bool rayIntersectsBVHNode(const Ray& ray, const BVHNodeGlslFormat& node) {
	// Based on https://gamedev.stackexchange.com/questions/18436/most-efficient-aabb-vs-ray-collision-algorithms
	float t1 = (node.boundingBoxMinX - ray.origin.x) / ray.dir.x;
	float t2 = (node.boundingBoxMaxX - ray.origin.x) / ray.dir.x;
	float t3 = (node.boundingBoxMinY - ray.origin.y) / ray.dir.y;
	float t4 = (node.boundingBoxMaxY - ray.origin.y) / ray.dir.y;
	float t5 = (node.boundingBoxMinZ - ray.origin.z) / ray.dir.z;
	float t6 = (node.boundingBoxMaxZ - ray.origin.z) / ray.dir.z;

	float tmin = max(max(min(t1, t2), min(t3, t4)), min(t5, t6));
	float tmax = min(min(max(t1, t2), max(t3, t4)), max(t5, t6));

	return tmax >= 0 && tmin <= tmax;
}

__device__ bool intersectRayAndSphere(const Ray& ray, const Sphere& s, RayHit* hit)
{
	float3 co = ray.origin - s.center;
	float b = 2 * dot(co, ray.dir);
	float c = dot(co, co) - s.radius*s.radius;
	float delta = b*b - 4 * c; // Since a is 1 (unitDirection dot unitDirection)

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

__device__ float3 getNormalAwayFromRay(Ray ray, Triangle t) {
	float3 normal = cross(t.b - t.a, t.c - t.a);

	// Avoiding branching - this will flip the sign of the normal if the normal forms an obtuse
	// angle with the direction of the ray
	return normalize(normal*dot(normal, -ray.dir));
}

__device__ bool isBVHNodeLeaf(const BVHNodeGlslFormat& node) {
	return 0 != (node.leftNodeOrObjectCount & (1 << 31));
}

// Moller–Trumbore intersection algorithm
// Source: https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
__device__ bool intersectRayAndTriangle(const Ray& ray, const Triangle& t, RayHit* hit)
{
	float3 edge1, edge2, h, s, q;
	float a, f, u, v;
	edge1 = (t.b - t.a);
	edge2 = (t.c - t.a);
	h = cross(ray.dir, edge2);
	a = dot(edge1, h);
	if (a > -EPSILON && a < EPSILON)
	{
		// This ray is parallel to this triangle.
		return false;
	}

	f = 1.0 / a;
	s = ray.origin - t.a;
	u = f * dot(s, h);
	if (u < 0.0 || u > 1.0)
	{
		return false;
	}

	q = cross(s, edge1);
	v = f * dot(ray.dir, q);
	if (v < 0.0 || u + v > 1.0)
	{
		return false;
	}

	// At this stage we can compute d to find out where the intersection point is on the line.
	float d = f * dot(edge2, q);
	if (d > EPSILON)
	{
		// ray intersection
		hit->pointOfHit = ray.pointAtDistance(d);
		hit->normal = getNormalAwayFromRay(ray, t);
		hit->material = t.material;
		return true;
	}
	else
	{
		// This means that there is a line intersection but not a ray intersection.
		return false;
	}
}

__device__ float lengthSquared(float3 v)
{
	return dot(v, v);
}

// __device__ bool castRay(const Ray& ray, const SceneInfo& scene, RayHit*  result) {
// 	int nodesIdxsToSearchStack[64];  // Given this is a binary search, 64 should be a sufficent size for
// 									 // the stack
// 	int stackSize = 1;
// 	nodesIdxsToSearchStack[0] = 0;
//
// 	float closestHitDistance = 1. / 0.;  // Infinity
//
//
// 	while (stackSize != 0) {
// 		int nodeIdx = nodesIdxsToSearchStack[stackSize - 1]; // pop off the top of the stack
// 		stackSize -= 1;
// 		BVHNodeGlslFormat node = scene.triangleBvhNodes[nodeIdx];
// 		if (rayIntersectsBVHNode(ray, node)) {
// 			if (isBVHNodeLeaf(node)) {
// 				for (int i = 0; i < (node.leftNodeOrObjectCount & 0x7fffffff); ++i) {
// 					Triangle triangle = scene.triangles[node.rightNodeOrObjectOffset + i];
//
// 					RayHit hit;
// 					if (intersectRayAndTriangle(ray, triangle, &hit)) {
// 						float distanceSquared = lengthSquared(ray.origin - hit.pointOfHit);
// 						if (closestHitDistance > distanceSquared) {
// 							closestHitDistance = distanceSquared;
// 							*result = hit;
// 						}
// 					}
// 				}
// 			}
// 			else {
// 				nodesIdxsToSearchStack[stackSize] = node.leftNodeOrObjectCount;
// 				stackSize += 1;
// 				nodesIdxsToSearchStack[stackSize] = node.rightNodeOrObjectOffset;
// 				stackSize += 1;
// 			}
// 		}
// 	}
//
// 	return !isinf(closestHitDistance);
// }


__device__ bool castRay(const Ray& ray, const SceneInfo& scene, RayHit*  result) {
	float closestHitDistance = 1e20;  // Infinity
									  //printf("%f %f %f, %f %f %f\n", ray.origin.x, ray.origin.y, ray.origin.z, ray.dir.x, ray.dir.y, ray.dir.z);

	RayHit tempResult;
	for (int i = 0; i < scene.trianglesCount; ++i)
	{
		if (intersectRayAndTriangle(ray, scene.triangles[i], &tempResult))
		{
			float distanceSquared = lengthSquared(ray.origin - tempResult.pointOfHit);
			if (closestHitDistance > distanceSquared)
			{
				closestHitDistance = distanceSquared;
				*result = tempResult;
			}
		}
	}

	for (int i = 0; i < scene.spheresCount; ++i)
	{
		if (intersectRayAndSphere(ray, scene.spheres[i], &tempResult))
		{
			float distanceSquared = lengthSquared(ray.origin - tempResult.pointOfHit);
			if (closestHitDistance > distanceSquared)
			{
				closestHitDistance = distanceSquared;
				*result = tempResult;
			}
		}
	}

	return closestHitDistance < 1e19;
}

__device__ float3 getHitIllumination(const SceneInfo& scene, const RayHit& hit) {
	float3 illumination = make_float3(0.2, 0.2, 0.2) * hit.material.color;
	for (int i = 0; i < scene.lightsCount; ++i)
	{
		float3 vectorToLight = scene.lights[i].position - hit.pointOfHit;

		RayHit hitTowardsLight;
		bool lightReached = !castRay(
			Ray::fromPoints(hit.pointOfHit + (hit.normal * EPSILON), scene.lights[i].position),
			scene, &hitTowardsLight);
		lightReached =
			lightReached || (
				lengthSquared(hitTowardsLight.pointOfHit - hit.pointOfHit) >
				lengthSquared(vectorToLight));
		if (lightReached)
		{
			illumination += hit.material.color * scene.lights[i].intensity * dot(normalize(vectorToLight), hit.normal);
		}
	}


	return illumination;
}

__device__ float3 getRayColor(const SceneInfo& scene, Ray ray) {
	const int MAX_RAY_BOUNCE = 20;

	float3 currentModifier = make_float3(1.0, 1.0, 1.0);

	for (int bounce = 0; bounce < MAX_RAY_BOUNCE; ++bounce)
	{
		RayHit hit;
		if (!castRay(ray, scene, &hit))
		{
			break;
		}

		if (hit.material.type == 1)
		{
			// Specular
			currentModifier *= hit.material.specularReflectivity;
			ray.origin = hit.pointOfHit + (hit.normal * EPSILON);
			ray.dir = reflect(ray.dir, hit.normal);

		}
		else if (hit.material.type == 2)
		{
			// // Refractive
			// if (dot(hit.normal, ray.dir) < 0) {
			// 	// Ray comes from the inside
			// 	currentModifier *= hit.material.specularReflectivity;
			// 	ray.origin = hit.pointOfHit + (-hit.normal * EPSILON);
			// 	ray.dir = refract(ray.dir, hit.normal, 1.0 / hit.material.refractiveIndex);
			// }
			// else {
			// 	// Ray comes from the outside
			// 	currentModifier *= v3(hit.material.specularReflectivity);
			// 	ray.origin = hit.pointOfHit + (hit.normal * EPSILON);
			// 	ray.dir = refract(ray.dir, -hit.normal, 1.0 / hit.material.refractiveIndex);
			// }
			return make_float3(1.0, 1.0, 0.0);
		}
		else
		{
			// Diffuse
			return currentModifier * getHitIllumination(scene, hit);
		}
	}

	return make_float3(0.0, 0.0, 0.0);
}


__global__ void renderingKernel(CudaCamera cam, SceneInfo scene, unsigned screenWidth,
	unsigned screenHeight, cudaSurfaceObject_t output)
{
	// Calculate surface coordinates
	unsigned int xPixel = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int yPixel = blockIdx.y * blockDim.y + threadIdx.y;

	xpix = xPixel;
	ypix = yPixel;

	Ray ray = getCameraRay(cam, xPixel / (float)screenWidth, yPixel / (float)screenHeight);

	uchar4 color;
	float3 colorFloat = getRayColor(scene, ray);

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
			renderingKernel<<<grid, block>>>(camera.getCudaInfo(), scene, outputWidth, outputHeight, viewCudaSurfaceObject);
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

SceneInfo SceneInfo::fromScene(const Scene& scene)
{
	SceneInfo result;

	std::vector<Triangle> cop = scene.triangles;
	auto bvh = buildBVH(std::move(cop));
	auto nodes = bvh->makeGpuBvh();

	cudaMalloc(&result.spheres, scene.spheres.size() * sizeof(Sphere));
	result.spheresCount = scene.spheres.size();
	cudaMemcpy(result.spheres, scene.spheres.data(), scene.spheres.size() * sizeof(Sphere), cudaMemcpyHostToDevice);

	cudaMalloc(&result.triangles, nodes.shapes.size() * sizeof(Triangle));
	result.trianglesCount = nodes.shapes.size();
	cudaMemcpy(result.triangles, nodes.shapes.data(), nodes.shapes.size() * sizeof(Triangle), cudaMemcpyHostToDevice);

	cudaMalloc(&result.lights, scene.lights.size() * sizeof(PointLightSource));
	result.lightsCount = scene.lights.size();
	cudaMemcpy(result.lights, scene.lights.data(), scene.lights.size() * sizeof(PointLightSource), cudaMemcpyHostToDevice);

	

	cudaMalloc(&result.triangleBvhNodes, nodes.bvhNodes.size() * sizeof(BVHNodeGlslFormat));
	cudaMemcpy(result.triangleBvhNodes, nodes.bvhNodes.data(), nodes.bvhNodes.size() * sizeof(BVHNodeGlslFormat), cudaMemcpyHostToDevice);

	return result;
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
