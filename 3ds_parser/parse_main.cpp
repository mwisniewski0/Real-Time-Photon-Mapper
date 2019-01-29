//
// Created by Beau Carlborg on 2019-01-28.
//

#include "parse_main.h"
#include <stdio.h>
#include <zconf.h>
#include <cstdlib>


Lib3dsFile* read_3ds_file(const char* file_path) {
    Lib3dsFile *parsed_file = lib3ds_file_open(file_path);

    if (!parsed_file) {
        printf("****ERROR*****");
        printf("Failed to open file %s\n", file_path);
        return 0;
    }

    return parsed_file;
}

int calculate_total_triangles(Lib3dsFile *f) {
    int i, j, total_faces = 0;

    for (i = 0; i < f->nmeshes; i++) {
        Lib3dsMesh *curr_mesh = f->meshes[i];
        total_faces += curr_mesh->nfaces;
    }

    return total_faces;
}


int calculate_total_verticies(Lib3dsFile *f) {
    int i, j, total_verticies = 0;

    for (i = 0; i < f->nmeshes; i++) {
        Lib3dsMesh *curr_mesh = f->meshes[i];
        total_verticies += curr_mesh->nvertices;
    }

    return total_verticies;
}


typedef struct flat_triangle {
    float verticies[3][4];
} triangle;


void copy_verticies(float *src, float *dst) {
    int i;

    for (i = 0; i < 3; i++) {
        dst[i] = src[i];
    }
}

flat_triangle *create_flat_geometry_array(Lib3dsFile *f, int total_triangles_count) {
    int i, j, k, curr_flat_triangle_index = 0;

    flat_triangle *flat_triangle_array = (flat_triangle *) malloc(sizeof(triangle) * total_triangles_count);

    for (i = 0; i < f->nmeshes; i++) {
        Lib3dsMesh *curr_mesh = f->meshes[i];


        for (j=0; j<curr_mesh->nfaces; j++) {
            Lib3dsFace curr_face = curr_mesh->faces[j];

            for (k=0; k<3; k++) {
                int src_verticie_index = curr_face.index[k];
                float *src_verticie = curr_mesh->vertices[src_verticie_index];

                float *dst_verticie = flat_triangle_array[curr_flat_triangle_index].verticies[k];
                copy_verticies(src_verticie, dst_verticie);
            }

            curr_flat_triangle_index++;
        }
    }
    return flat_triangle_array;
}


int main(int argc, char **argv) {
    Lib3dsFile *new_file = read_3ds_file(argv[1]);

    int total_triangles = calculate_total_triangles(new_file);
    int total_verticies = calculate_total_verticies(new_file);

    flat_triangle *triangles_array = create_flat_geometry_array(new_file, total_triangles);


    return 0;
}