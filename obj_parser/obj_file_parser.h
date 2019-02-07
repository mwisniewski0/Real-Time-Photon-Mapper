//
// Created by Beau Carlborg on 2019-01-30.
//


#define TINYOBJLOADER_IMPLEMENTATION ""
#include "tiny_obj_loader.h"

#include "beau_geometry.h"



// returns a new triangle struct for the current face being read from the obj file
// vertices -- a pointer to the vector of all of the vetices in the scene
// face_base_index -- the index of the x float for the first vertex in the current fact
// material_id for the current face
//


void copy_textures(Triangle *new_triangle,
                   tinyobj::index_t v0_index,
                   tinyobj::index_t v1_index,
                   tinyobj::index_t v2_index,
                   std::vector<tinyobj::real_t> *texcoords
                   );



void copy_vertices(Triangle *new_triangle,
                   tinyobj::index_t v0_index,
                   tinyobj::index_t v1_index,
                   tinyobj::index_t v2_index,
                   std::vector<tinyobj::real_t> *vertices
                   );


Triangle create_triangle(tinyobj::index_t face_base_index,
                         int material_index,
                         std::vector<tinyobj::index_t> face_vertex_indices,
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
                           std::vector<Triangle> *triangles);
