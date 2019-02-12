#include <cstdio>
#include <string>
#include <tiny_obj_loader.h>

int main(int argc, char **argv) {

	//    Values that will be filled by call to tinyobj::LoadObj
	auto *attrib = new tinyobj::attrib_t;
	auto *shapes = new std::vector<tinyobj::shape_t>;
	auto *materials = new std::vector<tinyobj::material_t>;
	std::string warn;
	std::string err;
	const char *filename = "/Users/beaucarlborg/CLionProjects/Real-Time-Photon-Mapper/3ds_test_files/lamborginhi/Lamborghini_Aventador.obj";
	const char *mtl_basedir = "/Users/beaucarlborg/CLionProjects/Real-Time-Photon-Mapper/3ds_test_files/lamborginhi";
	std::string texture_basedir = "/Users/beaucarlborg/CLionProjects/Real-Time-Photon-Mapper/3ds_test_files/lamborginhi";

	//    Reading obj and mat file


	printf("above build bvh\n");
	// buildBVH(std::move(triangles));
	printf("below build bvh\n");


	return 0;
}
