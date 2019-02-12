//
// Created by Beau Carlborg on 2019-01-30.
//

#include "obj_file_parser.h"

#include "../common/ply.h"

#include <stdio.h>
#include <iostream>



void build_texcoords(float3 *vertex_texcoords, tinyobj::index_t v0_index, tinyobj::index_t v1_index,
        tinyobj::index_t v2_index, std::vector<tinyobj::real_t> *texcoords) {

    float v0v = texcoords->at((2 * v0_index.texcoord_index));
    float v0t = texcoords->at((2 * v0_index.texcoord_index) + 1);

    vertex_texcoords[0] = make_float3(v0v, v0t, 0.0);

    float v1v = texcoords->at((2 * v1_index.texcoord_index));
    float v1t = texcoords->at((2 * v1_index.texcoord_index) + 1);

    vertex_texcoords[1] = make_float3(v1v, v1t, 0.0);

    float v2v = texcoords->at((2 * v2_index.texcoord_index));
    float v2t = texcoords->at((2 * v2_index.texcoord_index) + 1);

    vertex_texcoords[2] = make_float3(v2v, v2t, 0.0);
}



 void build_vertices(float3 *triangle_vertices, tinyobj::index_t v0_index, tinyobj::index_t v1_index,
        tinyobj::index_t v2_index, std::vector<tinyobj::real_t> *vertices) {

    float v0x = vertices->at((3 * v0_index.vertex_index));
    float v0y = vertices->at((3 * v0_index.vertex_index) + 1);
    float v0z = vertices->at((3 * v0_index.vertex_index) + 2);

     triangle_vertices[0] = make_float3(v0x, v0y, v0z);

    float v1x = vertices->at((3 * v1_index.vertex_index));
    float v1y = vertices->at((3 * v1_index.vertex_index) + 1);
    float v1z = vertices->at((3 * v1_index.vertex_index) + 2);

     triangle_vertices[1] = make_float3(v1x, v1y, v1z);

    float v2x = vertices->at((3 * v2_index.vertex_index));
    float v2y = vertices->at((3 * v2_index.vertex_index) + 1);
    float v2z = vertices->at((3 * v2_index.vertex_index) + 2);

     triangle_vertices[2] = make_float3(v2x, v2y, v2z);

}


void construct_material(Material *material_dst, tinyobj::material_t *material_src, std::string *material_base_path) {
    material_dst->diffuse = make_float3(material_src->diffuse[0], material_src->diffuse[1], material_src->diffuse[2]);
    material_dst->specular = make_float3(material_src->specular[0], material_src->specular[1], material_src->specular[2]);
    material_dst->transmittance = make_float3(material_src->transmittance[0], material_src->transmittance[1], material_src->transmittance[2]);
    material_dst->shininess = material_src->shininess;
    material_dst->refractiveIndex = material_src->ior;
    material_dst->diffuseTexture = GPUTexture::fromPng(*material_base_path + material_src->diffuse_texname);
}



Triangle create_triangle(tinyobj::index_t v0_index, tinyobj::index_t v1_index, tinyobj::index_t v2_index,
        int material_index, tinyobj::attrib_t *attrib, std::vector<tinyobj::material_t> *materials, unsigned long triangle_index, std::string *texture_base_path) {


    float3 triangle_verts[3];
    build_vertices(triangle_verts, v0_index, v1_index, v2_index, &(attrib->vertices));

    Material new_mat = {};

    construct_material(&new_mat, &(materials->at(material_index)), texture_base_path);

    Triangle new_triangle = Triangle::from3Points(triangle_verts[0], triangle_verts[1], triangle_verts[2], new_mat);

    float3 triangle_vert_texcoords[3];
    build_texcoords(triangle_vert_texcoords, v0_index, v1_index, v2_index, &(attrib->vertices));

    new_triangle.v0vt = triangle_vert_texcoords[0];
    new_triangle.v1vt = triangle_vert_texcoords[1];
    new_triangle.v2vt = triangle_vert_texcoords[2];

    return new_triangle;
}



void add_mesh_triangles(tinyobj::mesh_t *mesh, tinyobj::attrib_t *attrib, std::vector<tinyobj::material_t> *materials,
        std::vector<Triangle> *triangles, std::string *texture_base_path) {

//  for triangle face in shape
    for (int i = 0, j = 0; i < mesh->indices.size(); i += 3, ++j) {
        tinyobj::index_t v1_index = mesh->indices.at(i);
        tinyobj::index_t v2_index = mesh->indices.at(i);
        tinyobj::index_t v3_index = mesh->indices.at(i);

        int material_index = mesh->material_ids.at(j);

        Triangle new_triangle = create_triangle(v1_index, v2_index, v3_index, material_index, attrib, materials, j, texture_base_path);
        triangles->push_back(new_triangle);

    }
}



void build_triangles_array(std::vector<tinyobj::shape_t> *shapes, tinyobj::attrib_t *attrib, std::vector<tinyobj::material_t> *materials,
        std::vector<Triangle> *triangles, std::string *texture_base_path) {

//  for each shape in the scene
    for (int i = 0; i < shapes->size(); ++i) {
        tinyobj::shape_t curr = shapes->at(i);
        tinyobj::mesh_t mesh = shapes->at(i).mesh;
        add_mesh_triangles(&(mesh), attrib, materials, triangles, texture_base_path);

    }
}


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
    bool successful_obj_load = tinyobj::LoadObj(attrib, shapes, materials, &warn, &err, filename, mtl_basedir);
    if (!successful_obj_load) {
        std::cout << "obj file was unsuccessfully read!" << std::endl;
        return 1;
    }

    std::vector<Triangle> triangles;
    build_triangles_array(shapes, attrib, materials, &triangles, &texture_basedir);


    printf("above build bvh\n");
    buildBVH(std::move(triangles));
    printf("below build bvh\n");


    return 0;
}
