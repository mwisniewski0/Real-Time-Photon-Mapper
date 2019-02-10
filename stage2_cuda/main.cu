#include <iostream>
#include "Renderer.h"


int main(int argc, char* argv[])
{
	try {
		// TODO(#4): This should come from flags
		RendererConfig config;
		config.outputHeight = 300;
		config.outputWidth = 400;
		config.fragShaderPath = "./tracer.frag";
		config.vtxShaderPath = "./tracer.vert";
		config.windowTitle = "Photon";
		config.horizontalFovDegrees = 50.f;

		Renderer renderer(config);
		renderer.run();
		std::cin.ignore();
	}
	catch (std::runtime_error err)
	{
		std::cout << err.what();
		std::cin.ignore();
		return 1;
	}

	return 0;
}
