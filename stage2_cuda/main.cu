#include <iostream>
#include <string>
#include "Renderer.h"
#include "../external/cxxopts.hpp"


int main(int argc, char* argv[])
{
	cxxopts::Options options("PhotonS2", "Stage 2 of Photon for real time rendering of .photon files");
	options.add_options()
		("w,width", "Width of the output window", cxxopts::value<unsigned>()->default_value("400"))
		("h,height", "Height of the output window", cxxopts::value<unsigned>()->default_value("300"))
		("f,hfov", "Horizontal field of view in degrees", cxxopts::value<float>()->default_value("40.0"))
		("i,input", "Path to the photon file", cxxopts::value<std::string>())
		("help", "Print help");

	try {
		auto parsed = options.parse(argc, argv);

		if (parsed.count("help"))
		{
			std::cout << options.help() << std::endl;
			return 0;
		}
		if (parsed.count("input") == 0)
		{
			std::cout << "Missing --input flag. Try --help to learn more." << std::endl;
			return 1;
		}

		if (parsed.count("input") > 1)
		{
			std::cout << "Multiple input files specified. Try --help to learn more." << std::endl;
			return 1;
		}

		RendererConfig config;
		config.outputHeight = parsed["height"].as<unsigned>();
		config.outputWidth = parsed["width"].as<unsigned>();
		config.windowTitle = "Photon";
		config.horizontalFovDegrees = parsed["hfov"].as<float>();
		config.inputFile = parsed["input"].as<std::string>();

		Renderer renderer(config);
		renderer.run();
	}
	catch (std::exception err)
	{
		std::cerr << err.what();
		return 1;
	}
	catch (cxxopts::OptionException err)
	{
		std::cerr << err.what();
		return 1;
	}

	return 0;
}
