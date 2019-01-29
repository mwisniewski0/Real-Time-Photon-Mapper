#ifndef PROGRAM_H_INCLUDED__
#define PROGRAM_H_INCLUDED__

#include <string>
#include <SDL2/SDL.h>

// OpenGL / glew Headers
#define GL3_PROTOTYPES 1
#include <GL/glew.h>
#include "Camera.h"
#include <memory>

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
public:
	Renderer(const RendererConfig& config);
	void run();
};

#endif
