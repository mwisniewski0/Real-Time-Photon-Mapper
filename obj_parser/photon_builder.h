//
// Created by Beau Carlborg on 2019-02-09.
//

#ifndef PHOTON_PHOTON_BUILDER_H
#define PHOTON_PHOTON_BUILDER_H


//
// Created by Beau Carlborg on 2019-02-09.
//


#include "../common/geometry.h"
#include "../common/cutil_math.h"



//
// struct of the material that will be written to the photon file
// implements an array of floats rather than float3
//
typedef struct Binary_Material
{
    float diffuse[3];
    float specular[3];
    float transmittance[3];
    float shininess;
    int type;
} Binary_Material;



//
// struct of the triangle that will be written to the photon file
// implements arrays of floats rather than float3s
//
typedef struct Binary_Triangle
{
    float p[3];
    float v0[3];
    float v1[3];
    float normal[3];

    float v0vt[3];
    float v1vt[3];
    float v2vt[3];

    Binary_Material bin_material;
} Binary_Triangle;



//
// Creates a Binary_Material struct from a regular Material struct which
// can then be written to Binary_Triangles in the photon file.
//
Binary_Material construct_binary_material(Material material_in);



//
// Creates a regular material struct from the Binary_material struct
// that will be read from Binary_Triangle in the photon file
//
Material deconstruct_binary_material(Binary_Material material_in);



//
// Creates a Binary_Triangle struct from the Triangle struct
// The new struct can then be written to a photon file
//
Binary_Triangle construct_binary_triangle(Triangle triangle_in);



//
// Creates a regular Triangle struct from the Binary_Triangle struct
// that will be read from photon file
//
Triangle deconstruct_binary_triangle(Binary_Triangle triangle_in);



//
// Writes a vector of triangles to a photon file
//
void triangle_write(std::vector<Triangle> *triangles);



//
// Reads a photon file and appends triangles to the triangles vector argument
//
void triangle_read(std::vector<Triangle> *triangles);


#endif //PHOTON_PHOTON_BUILDER_H
