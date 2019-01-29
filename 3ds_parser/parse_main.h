//
// Created by Beau Carlborg on 2019-01-28.
//


#include <lib3ds.h>

#ifndef _parse3ds_main_h
#define _parse3ds_main_h

// Takes a path to a 3ds file and will parse it into a lib3ds file struct
Lib3dsFile* read_3ds_file(const char* file_path);

// Creates an array of all triangles in the scene, where each triangle is 3 verticies and each verticie is four floats
// Therefore the array of all triangles will look something like this

// float_x_v1, float_y_v1, float_z_v1, float_junk_v1, float_x_v2, float_y_v2, float_z_v2, float_junk_v2, float_x_v3, float_y_v3, float_z_v3, float_junk_v3

float *create_flat_geometry_array(Lib3dsFile *f, int total_triangles_count, int total_verticies_count);

int calculate_total_triangles(Lib3dsFile *f);

int calculate_total_verticies(Lib3dsFile *f);

void copy_verticies(float *src, float *dst);


// TODO: get total list of verticies
//
//float *generate_verticies_array(Lib3dsFile *f);





#endif
