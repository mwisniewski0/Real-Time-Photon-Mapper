#ifndef CUDA_RENDERER_H_INCLUDED__
#define CUDA_RENDERER_H_INCLUDED__

#ifdef _WIN64
#include <windows.h>
#endif

#include <GL/glew.h>
#include "cuda_gl_interop.h"
#include "../common/geometry.h"
#include "Camera.h"
#include "../cuda_common/cudaHelpers.h"
#include "../common/scene.h"
#include "../cuda_common/gpuBvh.h"
#include "../cuda_common/gpuScene.h"

// CudaRender is responsible for endering a ray traced scene using CUDA into an OpenGL
// texture.
class CudaRenderer
{
	SceneInfo scene;
	GLuint viewGLTexture;
	cudaGraphicsResource_t viewCudaResource;
	int outputWidth;
	int outputHeight;

	void initialize();
	
public:
	// Creates a new rendered with the provided output texture size
	CudaRenderer(int outputWidth, int outputHeight);

	// Loads the given scene into the GPU memory
	void loadScene(const Scene& scene);

	// Renders a single frame of the scene given the specified camera information
	void renderFrame(const Camera& camera);
};


#endif