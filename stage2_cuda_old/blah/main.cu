#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include "Renderer.h"


int main(int argc, char* argv[])
{
	try {
		// TODO(#4): This should come from flags
		RendererConfig config;
		config.outputHeight = 1080;
		config.outputWidth = 1920;
		config.fragShaderPath = "./tracer.frag";
		config.vtxShaderPath = "./tracer.vert";
		config.windowTitle = "Photon";
		config.horizontalFovDegrees = 50.f;

		Renderer renderer(config);
		renderer.run();
	}
	catch (std::runtime_error err)
	{
		std::cout << err.what();
		std::cin.ignore();
		return 1;
	}

	return 0;
}
