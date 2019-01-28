#include <iostream>
#include "Renderer.h"


int main(int argc, char* argv[])
{
	try {
		// TODO(#4): This should come from flags
		RendererConfig config;
		config.outputHeight = 600;
		config.outputWidth = 800;
		config.fragShaderPath = "./tracer.frag";
		config.vtxShaderPath = "./tracer.vert";
		config.windowTitle = "Photon";
		config.horizontalFovDegrees = 75.f;

		Renderer renderer(config);
		renderer.run();
	}
	catch (std::runtime_error err)
	{
		std::cout << err.what();
		return 1;
	}

	return 0;
}
