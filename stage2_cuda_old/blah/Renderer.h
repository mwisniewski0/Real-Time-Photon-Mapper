#ifndef PROGRAM_H_INCLUDED__
#define PROGRAM_H_INCLUDED__

#include <string>
#include <SDL2/SDL.h>

// OpenGL / glew Headers
#define GL3_PROTOTYPES 1
#include <GL/glew.h>
#include "Camera.h"
#include <memory>
#include <vector>

struct RendererConfig
{
	std::string inputFile;
	std::string vtxShaderPath;
	std::string fragShaderPath;
	std::string windowTitle;
	int outputWidth;
	int outputHeight;
	float horizontalFovDegrees;
};

struct UserMovementInfo
{
	bool movingForward = false;
	bool movingBackward = false;
	bool movingLeft = false;
	bool movingRight = false;

	void updateCamera(Camera* camera);
};

class FpsCounter
{
	int framesRendered = 0;
	int lastFrame = 0;

public:
	void secondPassed();
	void frameRendered();
};

class Renderer
{
	RendererConfig config;

	SDL_Window* mainWindow;
	SDL_GLContext mainContext;
	GLuint glProgramId;

	std::unique_ptr<Camera> camera;
	UserMovementInfo userInputState;
	FpsCounter fpsCounter;

	void pushCameraInfoToGPU();
	void initGL();
	void loadModel();
	void renderFrame();
	void loop();
	void cleanup();

	template <typename T>
	void addSsbo(const std::vector<T>& v, int layoutId)
	{
		GLuint ssbo;
		glGenBuffers(1, &ssbo);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
		glBufferData(GL_SHADER_STORAGE_BUFFER, v.size() * sizeof(T), &v[0], GL_DYNAMIC_DRAW);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, layoutId, ssbo);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); // unbind
	}

public:
	Renderer(const RendererConfig& config);
	void run();
};

#endif
