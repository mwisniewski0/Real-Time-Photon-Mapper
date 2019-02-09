//
// Created by Beau Carlborg on 2019-01-30.
//


#define TINYOBJLOADER_IMPLEMENTATION ""
#include "tiny_obj_loader.h"
//
//#include "beau_geometry.h"

#include "../common/geometry.h"



// returns a new triangle struct for the current face being read from the obj file
// vertices -- a pointer to the vector of all of the vetices in the scene
// face_base_index -- the index of the x float for the first vertex in the current fact
// material_id for the current face
//


float3 *build_texcoords(tinyobj::index_t v0_index,
                       tinyobj::index_t v1_index,
                       tinyobj::index_t v2_index,
                       std::vector<tinyobj::real_t> *texcoords
);


float3 *build_vertices(tinyobj::index_t v0_index,
                       tinyobj::index_t v1_index,
                       tinyobj::index_t v2_index,
                       std::vector<tinyobj::real_t> *vertices
);


Triangle create_and_push_triangle(tinyobj::index_t v0_index,
                                  tinyobj::index_t v1_index,
                                  tinyobj::index_t v2_index,
                                  int material_index,
                                  tinyobj::attrib_t *attrib,
                                  std::vector<tinyobj::material_t> *materials
);



// for mesh in shape in shapes
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
