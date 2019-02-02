//
// Created by Beau Carlborg on 2019-01-30.
//

#include <stdio.h>

#define TINYOBJLOADER_IMPLEMENTATION "fuck youuuu"
#include "tiny_obj_loader.h"


int main(int argc, char **argv) {

    tinyobj::attrib_t *attrib = new tinyobj::attrib_t;
    std::vector<tinyobj::shape_t> *shapes = new std::vector<tinyobj::shape_t>;
    std::vector<tinyobj::material_t> *materials = new std::vector<tinyobj::material_t>;
    std::string *warn;
    std::string *err;
    const char *filename = "../3ds_test_files/jireh3d_free_kitchet.obj";

    bool my_bool = tinyobj::LoadObj(attrib, shapes, materials, warn, err, filename);

    printf("look! It read a fucking file! %d\n", my_bool);
}