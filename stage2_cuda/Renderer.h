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
#include "../common/geometry.h"
#include "../common/scene.h"

// Configuration of the renderer, used for creating a new Renderer.
struct RendererConfig
{
	// The .photon file to be rendered
	std::string inputFile;

	// Desired title of the render window
	std::string windowTitle;

	// Desired size of the render window
	int outputWidth;
	int outputHeight;

	// Desired horizontal field of view
	float horizontalFovDegrees;
};

// Stores information about user's current actions while interacting with the scene.
struct UserMovementInfo
{
	bool movingForward = false;
	bool movingBackward = false;
	bool movingLeft = false;
	bool movingRight = false;

	// Updates the camera given the information about user actions.
	void updateCamera(Camera* camera);
};

// Responsible for counting the number of frames rendered in a given second.
class FpsCounter
{
	int framesRendered = 0;
	int lastFrame = 0;

public:
	// Should be called every time a second has passed.
	void secondPassed();

	// Should be called every time a frame is rendered.
	void frameRendered();
};

// Renderer is responsible for maintaining a renderer window and tying all parts of the
// program together to render .photon files in real time.
class Renderer
{
	RendererConfig config;

	SDL_Window* mainWindow;
	SDL_GLContext mainContext;

	std::unique_ptr<Camera> camera;
	UserMovementInfo userInputState;
	FpsCounter fpsCounter;

	void initGL();
	Scene loadModel();
	void renderFrame();
	void loop();
	void cleanup();

public:
	// Creates a new renderer given the configuration options. See docs for RendererConfig
	// for more information.
	Renderer(const RendererConfig& config);

	// Enters the input-render loop. This function is blocking.
	void run();
};

#endif
