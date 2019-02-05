#ifndef CUDA_RENDERER_H_INCLUDED__
#define CUDA_RENDERER_H_INCLUDED__

#ifdef _WIN64
#include <windows.h>
#endif

#include <GL/glew.h>
#include "cuda_gl_interop.h"
#include "../common/geometry.h"
#include "Camera.h"
#include "cudaHelpers.h"
#include "BVH.h"
#include "../common/scene.h"

struct SceneInfo
{
	GPUVector<Sphere> spheres;
	GPUVector<PointLightSource> lights;
	BVHGpuData triangleBvh;

	static SceneInfo fromScene(const Scene& scene);
};

class CudaRenderer
{
	SceneInfo scene;
	GLuint viewGLTexture;
	cudaGraphicsResource_t viewCudaResource;
	int outputWidth;
	int outputHeight;

	void initialize();
	
public:
	CudaRenderer(int outputWidth, int outputHeight);
	void loadScene(const Scene& scene);
	void renderFrame(const Camera& camera);
};


#endif