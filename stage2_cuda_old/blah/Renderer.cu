#include "Renderer.h"
#include "Camera.h"
#include "Helpers.h"
#include "Geometry.h"
#include <iostream>
#include <glm/glm.hpp>
#include "BVH.h"
#include "ply.h"


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

void Renderer::pushCameraInfoToGPU()
{
	camera->getGlslInfo().setInGlsl(glProgramId);
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
	SDL_GL_SetSwapInterval(1);

	glewExperimental = GL_TRUE;
	glewInit();

	glProgramId = glhelp::createShaderProgram(
		readFileToString(config.vtxShaderPath), readFileToString(config.fragShaderPath));
	glUseProgram(glProgramId);
}

void Renderer::loadModel()
{
	// TODO(#2): this function will load the model stored in config.inputFile. Currently, the model
	// is hardcoded (our model loading code is still in production).
	
//	std::vector<Triangle> triangles = loadTriangles("bun_zipper_res4.ply", Material{ {0,1,1}, {0.8f, 0.8f, 0.8f}, 2.5f, 0x0000 });

	std::vector<Triangle> triangles(12);
	// Back wall
	triangles[0].a = { -1, 1, 1 };
	triangles[0].b = { 1, 1, 1 };
	triangles[0].c = { -1, -1, 1 };
	triangles[0].material = {
		{ 1, 0, 0 }, // color
		{ 0.8f, 0.8f, 0.8f }, // reflectivity
		2.5,  // refractive index (diamond)
		0x0001 // type
	};
	triangles[1].a = { 1, 1, 1 };
	triangles[1].b = { 1, -1, 1 };
	triangles[1].c = { -1, -1, 1 };
	triangles[1].material = {
		{ 1, 0, 0 }, // color
		{ 0.8f, 0.8f, 0.8f }, // reflectivity
		2.5,  // refractive index (diamond)
		0x0001 // type
	};

	// Front wall
	triangles[2].a = { -1, 1, -4 };
	triangles[2].b = { 1, 1, -4 };
	triangles[2].c = { -1, -1, -4 };
	triangles[2].material = {
		{ 0, 1, 0 }, // color
		{ 0.8f, 0.8f, 0.8f }, // reflectivity
		2.5,  // refractive index (diamond)
		0x0000 // type
	};
	triangles[3].a = { 1, 1, -4 };
	triangles[3].b = { 1, -1, -4 };
	triangles[3].c = { -1, -1, -4 };
	triangles[3].material = {
		{ 0, 1, 0 }, // color
		{ 0.8f, 0.8f, 0.8f }, // reflectivity
		2.5,  // refractive index (diamond)
		0x0000 // type
	};

	// Left wall
	triangles[4].a = { -1, -1, 1 };
	triangles[4].b = { -1, 1, 1 };
	triangles[4].c = { -1, -1, -4 };
	triangles[4].material = {
		{ 0, 0, 1 }, // color
		{ 0.8f, 0.8f, 0.8f }, // reflectivity
		2.5,  // refractive index (diamond)
		0x0000 // type
	};
	triangles[5].a = { -1, 1, 1 };
	triangles[5].b = { -1, 1, -4 };
	triangles[5].c = { -1, -1, -4 };
	triangles[5].material = {
		{ 0, 0, 1 }, // color
		{ 0.8f, 0.8f, 0.8f }, // reflectivity
		2.5,  // refractive index (diamond)
		0x0000 // type
	};

	// Right wall
	triangles[6].a = { 1, -1, 1 };
	triangles[6].b = { 1, 1, 1 };
	triangles[6].c = { 1, -1, -4 };
	triangles[6].material = {
		{ 1, 1, 0 }, // color
		{ 0.8f, 0.8f, 0.8f }, // reflectivity
		2.5,  // refractive index (diamond)
		0x0000 // type
	};
	triangles[7].a = { 1, 1, 1 };
	triangles[7].b = { 1, 1, -4 };
	triangles[7].c = { 1, -1, -4 };
	triangles[7].material = {
		{ 1, 1, 0 }, // color
		{ 0.8f, 0.8f, 0.8f }, // reflectivity
		2.5,  // refractive index (diamond)
		0x0000 // type
	};

	// Top wall
	triangles[8].a = { -1, 1, 1 };
	triangles[8].b = { 1, 1, 1 };
	triangles[8].c = { -1, 1, -4 };
	triangles[8].material = {
		{ 0, 1, 1 }, // color
		{ 0.8f, 0.8f, 0.8f }, // reflectivity
		2.5,  // refractive index (diamond)
		0x0000 // type
	};

	triangles[9].a = { 1, 1, 1 };
	triangles[9].b = { 1, 1, -4 };
	triangles[9].c = { -1, 1, -4 };
	triangles[9].material = {
		{ 0, 1, 1 }, // color
		{ 0.8f, 0.8f, 0.8f }, // reflectivity
		2.5,  // refractive index (diamond)
		0x0000 // type
	};

	// Bottom wall
	triangles[10].a = { -1, -1, 1 };
	triangles[10].b = { 1, -1, 1 };
	triangles[10].c = { -1, -1, -4 };
	triangles[10].material = {
		{ 1, 0, 1 }, // color
		{ 0.8f, 0.8f, 0.8f }, // reflectivity
		2.5,  // refractive index (diamond)
		0x0000 // type
	};

	triangles[11].a = { 1, -1, 1 };
	triangles[11].b = { 1, -1, -4 };
	triangles[11].c = { -1, -1, -4 };
	triangles[11].material = {
		{ 1, 0, 1 }, // color
		{ 0.8f, 0.8f, 0.8f }, // reflectivity
		2.5,  // refractive index (diamond)
		0x0000 // type
	};
//
//	// prism
//	triangles[12].a = { 0, 0.5, 0 }; // A
//	triangles[12].b = { 0, 0, 0.5 }; // B
//	triangles[12].c = { -0.5, 0, 0 }; // C
//	triangles[12].material = {
//		{ 1, 1, 1 }, // color
//		{ 0.8f, 0.8f, 0.8f }, // reflectivity
//		2.5,  // refractive index (diamond)
//		0x0002 // type
//	};
//	triangles[13].a = { 0, 0.5, 0 }; // A
//	triangles[13].b = { -0.5, 0, 0 }; // C
//	triangles[13].c = { 0, 0, -0.5 }; // D
//	triangles[13].material = {
//		{ 1, 0, 1 }, // color
//		{ 0.8f, 0.8f, 0.8f }, // reflectivity
//		2.5,  // refractive index (diamond)
//		0x0002 // type
//	};
//	triangles[14].a = { 0, 0.5, 0 }; // A
//	triangles[14].b = { 0, 0, -0.5 }; // D
//	triangles[14].c = { 0.5, 0, 0 }; // E
//	triangles[14].material = {
//		{ 0, 0, 1 }, // color
//		{ 0.8f, 0.8f, 0.8f }, // reflectivity
//		2.5,  // refractive index (diamond)
//		0x0002 // type
//	};
//	triangles[15].a = { 0, 0.5, 0 }; // A
//	triangles[15].b = { 0.5, 0, 0 }; // E
//	triangles[15].c = { 0, 0, 0.5 }; // B
//	triangles[15].material = {
//		{ 1, 0, 0 }, // color
//		{ 0.8f, 0.8f, 0.8f }, // reflectivity
//		2.5,  // refractive index (diamond)
//		0x0002 // type
//	};
//
//
//	triangles[16].a = { 0, -0.5, 0 }; // F
//	triangles[16].b = { 0, 0, 0.5 }; // B
//	triangles[16].c = { -0.5, 0, 0 }; // C
//	triangles[16].material = {
//		{ 1, 0, 1 }, // color
//		{ 0.8f, 0.8f, 0.8f }, // reflectivity
//		2.5,  // refractive index (diamond)
//		0x0002 // type
//	};
//	triangles[17].a = { 0, -0.5, 0 }; // F
//	triangles[17].b = { -0.5, 0, 0 }; // C
//	triangles[17].c = { 0, 0, -0.5 }; // D
//	triangles[17].material = {
//		{ 1, 0, 1 }, // color
//		{ 0.8f, 0.8f, 0.8f }, // reflectivity
//		2.5,  // refractive index (diamond)
//		0x0002 // type
//	};
//	triangles[18].a = { 0, -0.5, 0 }; // F
//	triangles[18].b = { 0, 0, -0.5 }; // D
//	triangles[18].c = { 0.5, 0, 0 }; // E
//	triangles[18].material = {
//		{ 1, 0, 1 }, // color
//		{ 0.8f, 0.8f, 0.8f }, // reflectivity
//		2.5,  // refractive index (diamond)
//		0x0002 // type
//	};
//	triangles[19].a = { 0, -0.5, 0 }; // F
//	triangles[19].b = { 0.5, 0, 0 }; // E
//	triangles[19].c = { 0, 0, 0.5 }; // B
//	triangles[19].material = {
//		{ 1, 0, 1 }, // color
//		{ 0.8f, 0.8f, 0.8f }, // reflectivity
//		2.5,  // refractive index (diamond)
//		0x0002 // type
//	};

	auto bvh = buildBVH(std::move(triangles));
	auto gpuBvh = bvh->makeGpuBvh();

	Lighting light;
	light.lightSources = std::vector<PointLightSource>(1);
	light.lightSources[0].intensity = { 1, 1, 1 };
	light.lightSources[0].position = { 0, 0.85f, 0 };

	addSsbo(gpuBvh.bvhNodes, 4);
	addSsbo(light.lightSources, 3);
	addSsbo(gpuBvh.shapes, 2);
}

Renderer::Renderer(const RendererConfig& config)
{
	this->config = config;

	// TODO(#2): The input file should specify information about camera. This will be hardcoded for
	// now.
	this->camera = Camera::fromHorizontalFov(glm::vec3(0, 0, -1), glm::vec3(0, 0, 1),
		glm::vec3(0, 1, 0), glm::radians(config.horizontalFovDegrees),
		((float) config.outputWidth) / config.outputHeight);
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
				pushCameraInfoToGPU();
			}

			if (event.type == SDL_USEREVENT)
			{
				switch (event.user.code)
				{
				case UPDATE_TIMER:
					userInputState.updateCamera(camera.get());
					pushCameraInfoToGPU();
					break;
				case FPS_COUNTER_TIMER:
					fpsCounter.secondPassed();
					break;
				}
			}
		}
		renderFrame();
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
	loadModel();
	pushCameraInfoToGPU();
	loop();
	cleanup();
}
