#include "Renderer.h"
#include "Camera.h"
#include "Helpers.h"
#include "Geometry.h"
#include <glm/detail/func_trigonometric.inl>


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
	
	SceneGeometry geo;
	geo.sceneGeometry = std::vector<Triangle>(12);

	// Back wall
	geo.sceneGeometry[0].a = { -1, 1, 1 };
	geo.sceneGeometry[0].b = { 1, 1, 1 };
	geo.sceneGeometry[0].c = { -1, -1, 1 };
	geo.sceneGeometry[0].material = {
		{ 1, 0, 0 }, // color
		{ 0.8f, 0.8f, 0.8f }, // reflectivity
		0x0001 // type
	};
	geo.sceneGeometry[1].a = { 1, 1, 1 };
	geo.sceneGeometry[1].b = { 1, -1, 1 };
	geo.sceneGeometry[1].c = { -1, -1, 1 };
	geo.sceneGeometry[1].material = {
		{ 1, 0, 0 }, // color
		{ 0.8f, 0.8f, 0.8f }, // reflectivity
		0x0001 // type
	};

	// Front wall
	geo.sceneGeometry[2].a = { -1, 1, -1 };
	geo.sceneGeometry[2].b = { 1, 1, -1 };
	geo.sceneGeometry[2].c = { -1, -1, -1 };
	geo.sceneGeometry[2].material = {
		{ 0, 1, 0 }, // color
		{ 0.8f, 0.8f, 0.8f }, // reflectivity
		0x0000 // type
	};
	geo.sceneGeometry[3].a = { 1, 1, -1 };
	geo.sceneGeometry[3].b = { 1, -1, -1 };
	geo.sceneGeometry[3].c = { -1, -1, -1 };
	geo.sceneGeometry[3].material = {
		{ 0, 1, 0 }, // color
		{ 0.8f, 0.8f, 0.8f }, // reflectivity
		0x0000 // type
	};

	// Left wall
	geo.sceneGeometry[4].a = { -1, -1, 1 };
	geo.sceneGeometry[4].b = { -1, 1, 1 };
	geo.sceneGeometry[4].c = { -1, -1, -1 };
	geo.sceneGeometry[4].material = {
		{ 0, 0, 1 }, // color
		{ 0.8f, 0.8f, 0.8f }, // reflectivity
		0x0000 // type
	};
	geo.sceneGeometry[5].a = { -1, 1, 1 };
	geo.sceneGeometry[5].b = { -1, 1, -1 };
	geo.sceneGeometry[5].c = { -1, -1, -1 };
	geo.sceneGeometry[5].material = {
		{ 0, 0, 1 }, // color
		{ 0.8f, 0.8f, 0.8f }, // reflectivity
		0x0000 // type
	};

	// Right wall
	geo.sceneGeometry[6].a = { 1, -1, 1 };
	geo.sceneGeometry[6].b = { 1, 1, 1 };
	geo.sceneGeometry[6].c = { 1, -1, -1 };
	geo.sceneGeometry[6].material = {
		{ 1, 1, 0 }, // color
		{ 0.8f, 0.8f, 0.8f }, // reflectivity
		0x0000 // type
	};
	geo.sceneGeometry[7].a = { 1, 1, 1 };
	geo.sceneGeometry[7].b = { 1, 1, -1 };
	geo.sceneGeometry[7].c = { 1, -1, -1 };
	geo.sceneGeometry[7].material = {
		{ 1, 1, 0 }, // color
		{ 0.8f, 0.8f, 0.8f }, // reflectivity
		0x0000 // type
	};

	// Top wall
	geo.sceneGeometry[8].a = { -1, 1, 1 };
	geo.sceneGeometry[8].b = { 1, 1, 1 };
	geo.sceneGeometry[8].c = { -1, 1, -1 };
	geo.sceneGeometry[8].material = {
		{ 0, 1, 1 }, // color
		{ 0.8f, 0.8f, 0.8f }, // reflectivity
		0x0000 // type
	};

	geo.sceneGeometry[9].a = { 1, 1, 1 };
	geo.sceneGeometry[9].b = { 1, 1, -1 };
	geo.sceneGeometry[9].c = { -1, 1, -1 };
	geo.sceneGeometry[9].material = {
		{ 0, 1, 1 }, // color
		{ 0.8f, 0.8f, 0.8f }, // reflectivity
		0x0000 // type
	};

	// Bottom wall
	geo.sceneGeometry[10].a = { -1, -1, 1 };
	geo.sceneGeometry[10].b = { 1, -1, 1 };
	geo.sceneGeometry[10].c = { -1, -1, -1 };
	geo.sceneGeometry[10].material = {
		{ 1, 0, 1 }, // color
		{ 0.8f, 0.8f, 0.8f }, // reflectivity
		0x0000 // type
	};

	geo.sceneGeometry[11].a = { 1, -1, 1 };
	geo.sceneGeometry[11].b = { 1, -1, -1 };
	geo.sceneGeometry[11].c = { -1, -1, -1 };
	geo.sceneGeometry[11].material = {
		{ 1, 0, 1 }, // color
		{ 0.8f, 0.8f, 0.8f }, // reflectivity
		0x0000 // type
	};

	Lighting light;
	light.lightSources = std::vector<PointLightSource>(1);
	light.lightSources[0].intensity = { 1, 1, 1 };
	light.lightSources[0].position = { 0, 0.85f, 0 };

	GLuint geoSsbo;
	glGenBuffers(1, &geoSsbo);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, geoSsbo);
	glBufferData(GL_SHADER_STORAGE_BUFFER, geo.getSize(), geo.getPointer(), GL_DYNAMIC_DRAW);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, geoSsbo);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); // unbind

	GLuint lightSsbo;
	glGenBuffers(1, &lightSsbo);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, lightSsbo);
	glBufferData(GL_SHADER_STORAGE_BUFFER, light.getSize(), light.getPointer(), GL_DYNAMIC_DRAW);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, lightSsbo);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); // unbind
}

Renderer::Renderer(const RendererConfig& config)
{
	this->config = config;

	// TODO(#2): The input file should specify information about camera. This will be hardcoded for
	// now.
	this->camera = Camera::fromHorizontalFov(glm::vec3(0, -0.7, -0.7), glm::vec3(0, 0, 1),
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
					break;
				}
			}
		}
		renderFrame();
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
