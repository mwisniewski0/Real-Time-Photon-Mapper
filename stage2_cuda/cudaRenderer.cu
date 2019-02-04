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
#include "BVH.h"

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

// MollerTrumbore intersection algorithm
// Source: https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
__device__ bool intersectRayAndTriangle(const Ray& ray, const Triangle& t, RayHit* hit)
{
	float3 edge1, edge2, h, s, q;
	float a, f, u, v;
	edge1 = t.v0;
	edge2 = t.v1;
	h = cross(ray.dir, edge2);
	a = dot(edge1, h);
	if (a > -EPSILON && a < EPSILON)
	{
		// This ray is parallel to this triangle.
		return false;
	}

	f = 1.0 / a;
	s = ray.origin - t.p;
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

__device__ int intersectTriangles(const Ray& r_in, float& t, const BVHGpuData& bvhData) {
	int id = -1;
	int stack[64]; // its reasonable to assume this will be way bigger than neccesary
	int stack_idx = 1;
	stack[0] = 0;
	float d;
	float tb = -1e20; // large negative
	while (stack_idx)
	{
		int boxidx = stack[stack_idx - 1]; // pop off top of stack
		stack_idx--;
		if (!(bvhData.bvhNodes.contents[boxidx].u.leaf.count & 0x80000000))
		{
			// inner
			Box b;
			b.min = bvhData.bvhNodes.contents[boxidx].min;
			b.max = bvhData.bvhNodes.contents[boxidx].max;
			if (b.intersect(r_in))
			{
				stack[stack_idx++] = bvhData.bvhNodes.contents[boxidx].u.inner.left; // push right and left onto stack
				stack[stack_idx++] = bvhData.bvhNodes.contents[boxidx].u.inner.right;
			}
		}
		else
		{
			// leaf
			for (int i = bvhData.bvhNodes.contents[boxidx].u.leaf.offset;
			     i < bvhData.bvhNodes.contents[boxidx].u.leaf.offset + (bvhData.bvhNodes.contents[boxidx].u.leaf.count & 0x7fffffff);
			     i++)
			{
				// intersect all triangles in this box
				if ((d = bvhData.triangles.contents[i].intersect(r_in)) && d > -1e19)
				{

					if (d<t && d>0.001)
					{
						t = d;
						id = i;
					}
					else if (d>tb && d<0.001)
					{
						tb = d;
					}
				}
			}
		}
	}
	return id;
}

__device__ bool castRay2(const Ray& ray, const SceneInfo& scene, RayHit*  result) {
	float closestHitDistance = 1e20;  // Infinity
									  //printf("%f %f %f, %f %f %f\n", ray.origin.x, ray.origin.y, ray.origin.z, ray.dir.x, ray.dir.y, ray.dir.z);

	RayHit tempResult;
	// auto triangleIndex = intersectTriangles(ray, closestHitDistance, scene.triangleBvh);
	// if (triangleIndex >= 0)
	// {
	// 	result->pointOfHit = ray.pointAtDistance(closestHitDistance);
	// 	result->normal = getNormalAwayFromRay(ray, scene.triangleBvh.triangles.contents[triangleIndex]);
	// 	result->material = scene.triangleBvh.triangles.contents[triangleIndex].material;
	// }
	for (int i = 0; i < scene.triangleBvh.triangles.size; ++i)
	{
		if (intersectRayAndTriangle(ray, scene.triangleBvh.triangles.contents[i], &tempResult))
		{
			float distanceSquared = lengthSquared(ray.origin - tempResult.pointOfHit);
			if (closestHitDistance > distanceSquared)
			{
				closestHitDistance = distanceSquared;
				*result = tempResult;
			}
		}
	}

	for (int i = 0; i < scene.spheres.size; ++i)
	{
		if (intersectRayAndSphere(ray, scene.spheres.contents[i], &tempResult))
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
}__device__ bool castRay(const Ray& ray, const SceneInfo& scene, RayHit*  result) {
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

__device__ float3 getHitIllumination(const SceneInfo& scene, const RayHit& hit) {
	float3 illumination = make_float3(0.2, 0.2, 0.2) * hit.material.color;
	for (int i = 0; i < scene.lights.size; ++i)
	{
		float3 vectorToLight = scene.lights.contents[i].position - hit.pointOfHit;

		RayHit hitTowardsLight;
		bool lightReached = !castRay(
			Ray::fromPoints(hit.pointOfHit + (hit.normal * EPSILON), scene.lights.contents[i].position),
			scene, &hitTowardsLight);
		lightReached =
			lightReached || (
				lengthSquared(hitTowardsLight.pointOfHit - hit.pointOfHit) >
				lengthSquared(vectorToLight));
		if (lightReached)
		{
			illumination += hit.material.color * scene.lights.contents[i].intensity * dot(normalize(vectorToLight), hit.normal);
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
	std::unique_ptr<BVHNode> bvh = buildBVH(std::move(cop));
	result.triangleBvh = bvh->makeGpuBvh();

	result.spheres = vectorToGpu(scene.spheres);
	result.lights = vectorToGpu(scene.lights);

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
