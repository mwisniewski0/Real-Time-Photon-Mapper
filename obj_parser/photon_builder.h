//
// Created by Beau Carlborg on 2019-02-09.
//

#ifndef PHOTON_PHOTON_BUILDER_H
#define PHOTON_PHOTON_BUILDER_H


//
// Created by Beau Carlborg on 2019-02-09.
//

// C program for writing
// struct to file

#include "../common/geometry.h"
#include "../common/cutil_math.h"

typedef struct Binary_Material
{
    float diffuse[3];
    float specular[3];
    float transmittance[3];
    float shininess;
    int type;
} Binary_Material;

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


Binary_Material construct_binary_material(Material material_in);

Material deconstruct_binary_material(Binary_Material material_in);

Binary_Triangle construct_binary_triangle(Triangle triangle_in);

Triangle deconstruct_binary_triangle(Binary_Triangle triangle_in);


void triangle_write(std::vector<Triangle> *triangles);
void triangle_read(std::vector<Triangle> *triangles);
void person_write();
void person_read();
/*
 *
struct Triangle : public Shape {
	float3 p;  // v0
	float3 v0;   // v0->v1
	float3 v1;   // v0->v2
	float3 normal; //precompute and store. may not be faster needs testing
	Material material;

	float3 v0vt;
    float3 v1vt;
    float3 v2vt;
 *
 */



#endif //PHOTON_PHOTON_BUILDER_H
