//
// Created by Beau Carlborg on 2019-01-30.
//

#include <stdio.h>
#include <iostream>

#define TINYOBJLOADER_IMPLEMENTATION "fuck youuuu"
#include "tiny_obj_loader.h"

void copy_shape_vertices(std::vector<float> *output,  std::vector<tinyobj::real_t> *vrt_src, tinyobj::shape_t *shape) {
    std::vector<tinyobj::index_t> *indices = &(shape->mesh.indices);

    for (unsigned long i = 0; i < indices->size(); ++i) {
        auto curr_vert_index = (unsigned long) indices->at(i).vertex_index;
        unsigned long vert_base_index = curr_vert_index * 3;

        for (int j = 0; j < 3; j++) {
            tinyobj::real_t curr_val = vrt_src->at(vert_base_index + j);
            output->push_back((float) curr_val);
        }

        output->push_back(0.0);



//        printf("\t push vert <%f, %f, %f> --> <%f, %f, %f, %f>\n",
//                vrt_src->at(vert_base_index + 0), vrt_src->at(vert_base_index + 1), vrt_src->at(vert_base_index + 2),
//                output->at(output->size() - 4), output->at(output->size() - 3), output->at(output->size() - 2) , output->at(output->size() - 1)
//
//                );

    }
}

void fill_array_of_vertices(std::vector<float> *output,  std::vector<tinyobj::real_t> *vrt_src, std::vector<tinyobj::shape_t> *shapes) {

    for (unsigned long i = 0; i < shapes->size(); ++i) {
        std::cout << shapes->at(i).name << std::endl;
        copy_shape_vertices(output, vrt_src, &(shapes->at(i)));
    }
}


int main(int argc, char **argv) {

    auto *attrib = new tinyobj::attrib_t;
    auto *shapes = new std::vector<tinyobj::shape_t>;
    auto *materials = new std::vector<tinyobj::material_t>;
    std::string *warn;
    std::string *err;
    const char *filename = "/Users/beaucarlborg/CLionProjects/Real-Time-Photon-Mapper/3ds_test_files/kitchen.obj";

    bool my_bool = tinyobj::LoadObj(attrib, shapes, materials, warn, err, filename);


    auto *flattened_vert = new std::vector<float>;

    fill_array_of_vertices(flattened_vert, &(attrib->vertices), shapes);

}







