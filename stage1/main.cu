#include <iostream>
#include <string>
#ifdef _MSC_VER
#include <filesystem>
#else
#include <experimental/filesystem>
#endif
#include "../common/obj_file_parser.h"
#include "../external/cxxopts.hpp"
#include <fstream>


int main(int argc, char* argv[])
{
	cxxopts::Options options("PhotonS2", "Stage 2 of Photon for real time rendering of .photon files");
	options.add_options()
		("i,input", "Path to the obj file. Note that .mtl files and texture files are expected to be in the same directory", cxxopts::value<std::string>())
		("o,output", "Path to the output photon file", cxxopts::value<std::string>())
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
		
		if (parsed.count("output") == 0)
		{
			std::cout << "Missing --output flag. Try --help to learn more." << std::endl;
			return 1;
		}

		if (parsed.count("output") > 1)
		{
			std::cout << "Multiple output files specified. Try --help to learn more." << std::endl;
			return 1;
		}
		
		std::string objPath = parsed["input"].as<std::string>();
		std::string dirPath = std::experimental::filesystem::path(objPath).parent_path().generic_string();
		std::string outPath = parsed["output"].as<std::string>();

		auto scene = loadObj(objPath, dirPath, dirPath);

		std::ofstream outFile(outPath, std::ios_base::binary | std::ios_base::out);
		writeToStream(outFile, scene);
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
