#include "Renderer.h"
#include "Camera.h"
#include "../cuda_common/helpers.h"
#include "../common/geometry.h"
#include <iostream>
#include "../cuda_common/gpuBvh.h"
#include "../common/ply.h"
#include "cudaRenderer.h"
#include "GlHelp.h"
#include "sdlHelpers.h"
#include "../common/obj_file_parser.h"
#include <fstream>


// Photon-defined SDL messages
const auto UPDATE_TIMER = 100;
const auto FPS_COUNTER_TIMER = 101;


void UserMovementInfo::updateCamera(Camera* camera)
{
	if (movingForward)
	{
		camera->moveForward(0.02);
	}
	if (movingBackward)
	{
		camera->moveBackward(0.02);
	}
	if (movingLeft)
	{
		camera->moveLeft(0.02);
	}
	if (movingRight)
	{
		camera->moveRight(0.02);
	}
}

void FpsCounter::secondPassed()
{
	int FPS = framesRendered - lastFrame;
	std::cout << FPS << "\n";
	lastFrame = framesRendered;
}

void FpsCounter::frameRendered()
{
	framesRendered += 1;
}

void Renderer::initGL()
{
	// Initialize SDL's Video subsystem
	if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) < 0)
	{
		throw std::runtime_error("SDL could not be initialized");
	}

	mainWindow = SDL_CreateWindow(config.windowTitle.c_str(), SDL_WINDOWPOS_CENTERED,
		SDL_WINDOWPOS_CENTERED, config.outputWidth, config.outputHeight, SDL_WINDOW_OPENGL);
	if (!mainWindow)
	{
		throw std::runtime_error("Unable to create window");
	}

	// Create our opengl context and attach it to our window
	mainContext = SDL_GL_CreateContext(mainWindow);

	SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 6);

	// TODO: consider whether we want double buffering or not. We only draw one rectangle, so we
	// might not need the extra buffer.
	SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
	SDL_GL_SetSwapInterval(0);

	glewExperimental = GL_TRUE;
	glewInit();
}

struct Triangle2
{
	float3 a, b, c;
	unsigned materialIndex;

	Triangle toTriangle()
	{
		return Triangle::from3Points(a, b, c, materialIndex);
	}
};

Scene Renderer::loadModel()
{
	auto file = std::ifstream(config.inputFile, std::ios_base::binary | std::ios_base::in);
	auto scene = readFromStream<Scene>(file);

	// TODO: temporary add light sources and refractive sphere
	scene.lights.push_back(PointLightSource{
		{ 1.22f, 4.10f, -0.70f },
		{ 0.52f, 0.52f, 0.62f }
	});
	scene.lights.push_back(PointLightSource{
		{ 3.16f, 4.55f, -1.28f },
		{ 0.62f, 0.52f, 0.52f }
	});

	scene.spheres.push_back(Sphere{
		{ 0.207f, 1.656f, -3.006f },
		0.2f,
		{
			{ 0.0f,0.0f,0.0f },
			{ 0.0f,0.0f,0.0f },
			{ 0.9f, 0.9f, 0.9f },
			1.0f,
			1.6f,
			false,
			{}
		}
	});

	return scene;
}

float degToRadians(float a)
{
	return 0.017453292 * a;
}

Renderer::Renderer(const RendererConfig& config)
{
	this->config = config;

	// TODO(#2): The input file should specify information about camera. This will be hardcoded for
	// now.
	this->camera = Camera::fromHorizontalFov(make_float3(1.369, 3.40, -0.40), make_float3(1.37, -3.52, -8.94),
		make_float3(8.609e-05, 0.7772, -0.6296), degToRadians(config.horizontalFovDegrees),
		((float)config.outputWidth) / config.outputHeight);
}

void Renderer::renderFrame()
{
	glClearColor(0.5, 0.5, 0.5, 1);
	glClear(GL_COLOR_BUFFER_BIT);
	glhelp::drawSquare(0, 0, 2);
	SDL_GL_SwapWindow(mainWindow);
}

void Renderer::loop()
{
	CudaRenderer r(config.outputWidth, config.outputHeight);
	r.loadScene(loadModel());

	bool loop = true;

	// Update 20 times per second
	registerPeriodicalSDLMessage(1000. / 35., UPDATE_TIMER);
	registerPeriodicalSDLMessage(1000., FPS_COUNTER_TIMER);


	while (loop)
	{
		SDL_Event event;
		while (SDL_PollEvent(&event))
		{
			if (event.type == SDL_QUIT)
				loop = false;

			if (event.type == SDL_KEYDOWN)
			{
				switch (event.key.keysym.sym)
				{
				case SDLK_ESCAPE:
					loop = false;
					break;
				case SDLK_w:
					userInputState.movingForward = true;
					break;
				case SDLK_s:
					userInputState.movingBackward = true;
					break;
				case SDLK_a:
					userInputState.movingLeft = true;
					break;
				case SDLK_d:
					userInputState.movingRight = true;
					break;
				}
			}

			if (event.type == SDL_KEYUP)
			{
				switch (event.key.keysym.sym)
				{
				case SDLK_w:
					userInputState.movingForward = false;
					break;
				case SDLK_s:
					userInputState.movingBackward = false;
					break;
				case SDLK_a:
					userInputState.movingLeft = false;
					break;
				case SDLK_d:
					userInputState.movingRight = false;
					break;
				}
			}

			if (event.type == SDL_MOUSEMOTION)
			{
				SDL_SetRelativeMouseMode(SDL_TRUE);
				camera->absTurnRight(event.motion.xrel * 0.001);
				camera->absTurnUp(event.motion.yrel * 0.001);
			}

			if (event.type == SDL_USEREVENT)
			{
				switch (event.user.code)
				{
				case UPDATE_TIMER:
					userInputState.updateCamera(camera.get());
					break;
				case FPS_COUNTER_TIMER:
					fpsCounter.secondPassed();
					break;
				}
			}
		}

		r.renderFrame(*camera);
		SDL_GL_SwapWindow(mainWindow);
		fpsCounter.frameRendered();
	}
}

void Renderer::cleanup()
{
	SDL_GL_DeleteContext(mainContext);
	SDL_DestroyWindow(mainWindow);
	SDL_Quit();
}

void Renderer::run()
{
	initGL();
	loop();
	cleanup();
}
