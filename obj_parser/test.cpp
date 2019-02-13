#include <cstdio>
#include <string>
#include <tiny_obj_loader.h>
#include "obj_file_parser.h"

int main(int argc, char **argv) {

	//    Values that will be filled by call to tinyobj::LoadObj

	const char *filename = "/Users/beaucarlborg/CLionProjects/Real-Time-Photon-Mapper/3ds_test_files/lamborginhi/Lamborghini_Aventador.obj";
	const char *mtl_basedir = "/Users/beaucarlborg/CLionProjects/Real-Time-Photon-Mapper/3ds_test_files/lamborginhi";
	std::string texture_basedir = "/Users/beaucarlborg/CLionProjects/Real-Time-Photon-Mapper/3ds_test_files/lamborginhi";

	//    Reading obj and mat file
	loadObj(&filename, *mtl_basedir, texture_basedir);

	printf("above build bvh\n");
	// buildBVH(std::move(triangles));
	printf("below build bvh\n");


	return 0;
}
