//
// Created by Beau Carlborg on 2019-01-30.
//


#include "tiny_obj_loader.h"
//
//#include "beau_geometry.h"

#include "../common/geometry.h"
#include "../common/bvh.h"
#include "../cuda_common/gpuScene.h"


Scene loadObj(const std::string& objPath, const std::string& mtlBaseDir, const std::string& textureBaseDir);


// returns a new triangle struct for the current face being read from the obj file
// vertices -- a pointer to the vector of all of the vetices in the scene
// face_base_index -- the index of the x float for the first vertex in the current fact
// material_id for the current face
//


void build_texcoords(float3 *vertex_texcoords, tinyobj::index_t v0_index, tinyobj::index_t v1_index,
                     tinyobj::index_t v2_index, std::vector<tinyobj::real_t> *texcoords);


void build_vertices(float3 *triangle_vertices, tinyobj::index_t v0_index, tinyobj::index_t v1_index,
                    tinyobj::index_t v2_index, std::vector<tinyobj::real_t> *vertices);


void construct_material(Material *material_dst, tinyobj::material_t *material_src);


Triangle create_triangle(tinyobj::index_t v0_index, tinyobj::index_t v1_index, tinyobj::index_t v2_index,
                         int material_index, tinyobj::attrib_t *attrib, std::vector<tinyobj::material_t> *materials, unsigned long triangle_index);



void add_mesh_triangles(tinyobj::mesh_t *mesh,
                        tinyobj::attrib_t *attrib,
                        std::vector<tinyobj::material_t> *materials,
                        std::vector<Triangle> *triangles
);



void build_triangles_array(std::vector<tinyobj::shape_t> *shapes,
                           tinyobj::attrib_t *attrib,
                           std::vector<tinyobj::material_t> *materials,
                           std::vector<Triangle> *triangles
);
