//
// Created by Beau Carlborg on 2019-01-30.
//

#include "obj_file_parser.h"

#include <stdio.h>
#include <iostream>



Triangle create_triangle(tinyobj::index_t v1_index, tinyobj::index_t v2_index, tinyobj::index_t v3_index,
        int material_index,
        std::vector<tinyobj::real_t> *vertices,
        std::vector<tinyobj::material_t> *materials) {

    printf("----TRIANGLE\n");

    float3 v1 = make_float3(vertices->at((3 * v1_index.vertex_index)),
                            vertices->at((3 * v1_index.vertex_index) + 1),
                            vertices->at((3 * v1_index.vertex_index) + 2));


    float3 v2 = make_float3(vertices->at((3 * v2_index.vertex_index)),
                            vertices->at((3 * v2_index.vertex_index) + 1),
                            vertices->at((3 * v2_index.vertex_index) + 2));


    float3 v3 = make_float3(vertices->at((3 * v3_index.vertex_index)),
                            vertices->at((3 * v3_index.vertex_index) + 1),
                            vertices->at((3 * v3_index.vertex_index) + 2));



    // Can add whatever we want to materials here.
    Material *material = new Material;

    if (material_index != -1) {
        printf("----MATERIAL FOUND: INDEX %d\n", material_index);
    }

//    material->specularReflectivity = make_float3(materials->at(material_index).specular[0],
//                                                 materials->at(material_index).specular[1],
//                                                 materials->at(material_index).specular[2]);
//
//    material->refractiveIndex = materials->at(material_index).ior;


    return Triangle::from3Points(v1, v2, v3, *material);
}


void add_mesh_triangles(tinyobj::mesh_t *mesh,
        std::vector<tinyobj::real_t> *vertices,
        std::vector<tinyobj::material_t> *materials,
        std::vector<Triangle> *triangles) {

    for (int i = 0, j = 0; i < mesh->indices.size(); i += 3, ++j) {
        printf("----FACE\n");
        tinyobj::index_t v1_index = mesh->indices.at(i);
        tinyobj::index_t v2_index = mesh->indices.at(i);
        tinyobj::index_t v3_index = mesh->indices.at(i);

        int material_index = mesh->material_ids.at(j);

        Triangle new_triangle = create_triangle(v1_index, v2_index, v3_index, material_index, vertices, materials);
        triangles->push_back(new_triangle);
    }
}



void build_triangles_array(std::vector<tinyobj::shape_t> *shapes,
        std::vector<tinyobj::real_t> *vertices,
        std::vector<tinyobj::material_t> *materials,
        std::vector<Triangle> *triangles) {


//  for each shape in the scene
    for (int i = 0; i < shapes->size(); ++i) {
        printf("**************SHAPE****************\n");
        tinyobj::shape_t curr = shapes->at(i);
        tinyobj::mesh_t mesh = shapes->at(i).mesh;
        add_mesh_triangles(&(mesh), vertices, materials, triangles);
    }
}


int main(int argc, char **argv) {

    auto *attrib = new tinyobj::attrib_t;
    auto *shapes = new std::vector<tinyobj::shape_t>;
    auto *materials = new std::vector<tinyobj::material_t>;
    std::string *warn;
    std::string *err;
    const char *filename = "/Users/beaucarlborg/CLionProjects/Real-Time-Photon-Mapper/3ds_test_files/vonia/vonia.obj";
    const char *mtl_basedir = "/Users/beaucarlborg/CLionProjects/Real-Time-Photon-Mapper/3ds_test_files/vonia/materials/";

    bool successful_obj_load = tinyobj::LoadObj(attrib, shapes, materials, warn, err, filename, mtl_basedir);
    if (!successful_obj_load) {
        std::cout << "obj file was unsuccessfully read!" << std::endl;
        return 1;
    }

    // builds teh triangles
    std::vector<Triangle> triangles;

    // fills triangles vector
    build_triangles_array(shapes, &(attrib->vertices), materials, &triangles);

//    buildBVH(std::move(triangles));

//    'std::__1::vector<Triangle, std::__1::allocator<Triangle> >' to 'std::vector<Triangle> &&'

//    buildBVH(std::vector<Triangle>());


    return 0;
}

//

//typedef struct {
//    std::string name;
//
//    real_t ambient[3]; // no
//    real_t diffuse[3]; // yes
//    real_t specular[3];  // yes
//    real_t transmittance[3];  // yes
//    real_t emission[3];  // everything that has a non-zero value will become a light soure
//    real_t shininess;  // yes
//    real_t ior;       // type(ish)
//    real_t dissolve;  // 1 == opaque; 0 == fully transparent
//    // illumination model (see http://www.fileformat.info/format/material/)
//    int illum;
//}




