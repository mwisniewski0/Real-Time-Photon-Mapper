//
// Created by Beau Carlborg on 2019-01-30.
//


#define TINYOBJLOADER_IMPLEMENTATION ""
#include "tiny_obj_loader.h"

#include "../common/cutil_math.h"
#include "../common/geometry.h"d



// returns a new triangle struct for the current face being read from the obj file
// vertices -- a pointer to the vector of all of the vetices in the scene
// face_base_index -- the index of the x float for the first vertex in the current fact
// material_id for the current face
//

Triangle create_triangle(tinyobj::index_t face_base_index,
                         int material_index,
                         std::vector<tinyobj::index_t> face_vertex_indices,
                         std::vector<tinyobj::real_t> *vertices,
                         std::vector<tinyobj::material_t> *materials
                         );



// for mesh in shape in shapes
void add_mesh_triangles(tinyobj::mesh_t *mesh,
                        std::vector<tinyobj::real_t> *vertices,
                        std::vector<tinyobj::material_t> *materials,
                        std::vector<Triangle> *triangles
                        );




void build_triangles_array(std::vector<Triangle> *triangles,
                           std::vector<tinyobj::shape_t> *shapes,
                           std::vector<tinyobj::real_t> *vertices,
                           std::vector<tinyobj::material_t> *materials
                           );

// main