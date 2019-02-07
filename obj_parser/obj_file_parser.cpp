//
// Created by Beau Carlborg on 2019-01-30.
//

#include "obj_file_parser.h"

#include <stdio.h>
#include <iostream>

void copy_textures(Triangle *new_triangle, tinyobj::index_t v0_index, tinyobj::index_t v1_index,
                   tinyobj::index_t v2_index, std::vector<tinyobj::real_t> *texcoords) {

    new_triangle->v0vt[0] = texcoords->at((2 * v0_index.texcoord_index));
    new_triangle->v0vt[1] = texcoords->at((2 * v0_index.texcoord_index) + 1);

    new_triangle->v1vt[0] = texcoords->at((2 * v1_index.texcoord_index));
    new_triangle->v1[1] = texcoords->at((2 * v1_index.texcoord_index) + 1);

    new_triangle->v2vt[0] = texcoords->at((2 * v2_index.texcoord_index));
    new_triangle->v2vt[1] = texcoords->at((2 * v2_index.texcoord_index) + 1);

}



void copy_vertices(Triangle *new_triangle, tinyobj::index_t v0_index, tinyobj::index_t v1_index,
        tinyobj::index_t v2_index, std::vector<tinyobj::real_t> *vertices) {

    new_triangle->v0[0] = vertices->at((3 * v0_index.vertex_index));
    new_triangle->v0[1] = vertices->at((3 * v0_index.vertex_index) + 1);
    new_triangle->v0[2] = vertices->at((3 * v0_index.vertex_index) + 2);

    new_triangle->v1[0] = vertices->at((3 * v1_index.vertex_index));
    new_triangle->v1[1] = vertices->at((3 * v1_index.vertex_index) + 1);
    new_triangle->v1[2] = vertices->at((3 * v1_index.vertex_index) + 2);

    new_triangle->v2[0] = vertices->at((3 * v2_index.vertex_index));
    new_triangle->v2[1] = vertices->at((3 * v2_index.vertex_index) + 1);
    new_triangle->v2[2] = vertices->at((3 * v2_index.vertex_index) + 2);
}




Triangle create_triangle(tinyobj::index_t v0_index, tinyobj::index_t v1_index, tinyobj::index_t v2_index,
        int material_index,
        tinyobj::attrib_t *attrib,
        std::vector<tinyobj::material_t> *materials) {

    Triangle *new_triangle = new Triangle;

    // retrieving vertex locations: filling triangle.v0, triangle.v1, triangle.v2
    copy_vertices(new_triangle, v0_index, v1_index, v2_index, &(attrib->vertices));

    // retrieving texture vt coordinates: filling triangle.v0vt, triangle.v1vt, triangle.v2vt
    copy_textures(new_triangle, v0_index, v1_index, v2_index, &(attrib->texcoords));

    return *new_triangle;
}




void add_mesh_triangles(tinyobj::mesh_t *mesh,
        tinyobj::attrib_t *attrib,
        std::vector<tinyobj::material_t> *materials,
        std::vector<Triangle> *triangles) {

//    for triangle face in shape
    for (int i = 0, j = 0; i < mesh->indices.size(); i += 3, ++j) {
        tinyobj::index_t v1_index = mesh->indices.at(i);
        tinyobj::index_t v2_index = mesh->indices.at(i);
        tinyobj::index_t v3_index = mesh->indices.at(i);

        int material_index = mesh->material_ids.at(j);

        Triangle new_triangle = create_triangle(v1_index, v2_index, v3_index, material_index, attrib, materials);
        triangles->push_back(new_triangle);

    }
}



void build_triangles_array(std::vector<tinyobj::shape_t> *shapes,
        tinyobj::attrib_t *attrib,
        std::vector<tinyobj::material_t> *materials,
        std::vector<Triangle> *triangles) {


//  for each shape in the scene
    for (int i = 0; i < shapes->size(); ++i) {
        tinyobj::shape_t curr = shapes->at(i);
        tinyobj::mesh_t mesh = shapes->at(i).mesh;
        add_mesh_triangles(&(mesh), attrib, materials, triangles);

    }
}




int main(int argc, char **argv) {

//    Values that will be filled by call to tinyobj::LoadObj
    auto *attrib = new tinyobj::attrib_t;
    auto *shapes = new std::vector<tinyobj::shape_t>;
    auto *materials = new std::vector<tinyobj::material_t>;
    std::string *warn;
    std::string *err;
    const char *filename = "/Users/beaucarlborg/CLionProjects/Real-Time-Photon-Mapper/3ds_test_files/vonia/vonia.obj";
    const char *mtl_basedir = "/Users/beaucarlborg/CLionProjects/Real-Time-Photon-Mapper/3ds_test_files/vonia/materials/";

//    Reading obj and mat file
    bool successful_obj_load = tinyobj::LoadObj(attrib, shapes, materials, warn, err, filename, mtl_basedir);
    if (!successful_obj_load) {
        std::cout << "obj file was unsuccessfully read!" << std::endl;
        return 1;
    }

//    initializing geometry from results of tinyobj::LoadObj
    std::vector<Triangle> triangles;
    build_triangles_array(shapes, attrib, materials, &triangles);


//    buildBVH(std::move(triangles));
//    buildBVH(std::vector<Triangle>());

    return 0;
}
