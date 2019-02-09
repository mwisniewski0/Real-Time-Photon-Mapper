//
// Created by Beau Carlborg on 2019-02-09.
//

#include "./photon_builder.h"


#include <stdio.h>
#include <stdlib.h>
#include <string.h>


Binary_Material construct_binary_material(Material material_in) {
    Binary_Material bin_material;

    bin_material.diffuse[0] = material_in.diffuse.x;
    bin_material.diffuse[1] = material_in.diffuse.y;
    bin_material.diffuse[1] = material_in.diffuse.z;
    
    bin_material.specular[0] = material_in.specular.x;
    bin_material.specular[1] = material_in.specular.y;
    bin_material.specular[1] = material_in.specular.z;
    
    bin_material.transmittance[0] = material_in.transmittance.x;
    bin_material.transmittance[1] = material_in.transmittance.y;
    bin_material.transmittance[1] = material_in.transmittance.z;

    bin_material.shininess = material_in.shininess;

    bin_material.type = material_in.type;

    return bin_material;
}

Material deconstruct_binary_material(Binary_Material material_in) {
    Material material;

    material.diffuse = make_float3(material_in.diffuse[0],
                                   material_in.diffuse[1],
                                   material_in.diffuse[2]
    );

    material.specular = make_float3(material_in.specular[0],
                                   material_in.specular[1],
                                   material_in.specular[2]
    );

    material.transmittance = make_float3(material_in.transmittance[0],
                                   material_in.transmittance[1],
                                   material_in.transmittance[2]
    );

    material.shininess = material_in.shininess;

    material.type = material.type;

    return material;

}


Binary_Triangle construct_binary_triangle(Triangle triangle_in) {
    Binary_Triangle bin_triangle;

    bin_triangle.p[0] = triangle_in.p.x;
    bin_triangle.p[1] = triangle_in.p.y;
    bin_triangle.p[2] = triangle_in.p.z;

    bin_triangle.v0[0] = triangle_in.v0.x;
    bin_triangle.v0[1] = triangle_in.v0.y;
    bin_triangle.v0[2] = triangle_in.v0.z;

    bin_triangle.v1[0] = triangle_in.v1.x;
    bin_triangle.v1[1] = triangle_in.v1.y;
    bin_triangle.v1[2] = triangle_in.v1.z;

    bin_triangle.normal[0] = triangle_in.normal.x;
    bin_triangle.normal[1] = triangle_in.normal.y;
    bin_triangle.normal[2] = triangle_in.normal.z;

    bin_triangle.v0vt[0] = triangle_in.v0vt.x;
    bin_triangle.v0vt[1] = triangle_in.v0vt.y;
    bin_triangle.v0vt[2] = triangle_in.v0vt.z;

    bin_triangle.v1vt[0] = triangle_in.v1vt.x;
    bin_triangle.v1vt[1] = triangle_in.v1vt.y;
    bin_triangle.v1vt[2] = triangle_in.v1vt.z;

    bin_triangle.v2vt[0] = triangle_in.v2vt.x;
    bin_triangle.v2vt[1] = triangle_in.v2vt.y;
    bin_triangle.v2vt[2] = triangle_in.v2vt.z;

    bin_triangle.bin_material = construct_binary_material(triangle_in.material);

    return bin_triangle;
}


Triangle deconstruct_binary_triangle(Binary_Triangle triangle_in) {
    Triangle triangle;

    triangle.p = make_float3(triangle_in.p[0],
                             triangle_in.p[1],
                             triangle_in.p[2]
    );


    triangle.v0 = make_float3(triangle_in.v0[0],
                             triangle_in.v0[1],
                             triangle_in.v0[2]
    );

    triangle.v1 = make_float3(triangle_in.v1[0],
                             triangle_in.v1[1],
                             triangle_in.v1[2]
    );


    triangle.normal = make_float3(triangle_in.normal[0],
                             triangle_in.normal[1],
                             triangle_in.normal[2]
    );


    triangle.v0vt = make_float3(triangle_in.v0vt[0],
                             triangle_in.v0vt[1],
                             triangle_in.v0vt[2]
    );


    triangle.v1vt = make_float3(triangle_in.v1vt[0],
                             triangle_in.v1vt[1],
                             triangle_in.v1vt[2]
    );

    triangle.v2vt = make_float3(triangle_in.v2vt[0],
                             triangle_in.v2vt[1],
                             triangle_in.v2vt[2]
    );


    triangle.material = deconstruct_binary_material(triangle_in.bin_material);

    return triangle;
}




void triangle_write(std::vector<Triangle> *triangles) {
    FILE *outfile;

    // open file for writing
    outfile = fopen ("/Users/beaucarlborg/CLionProjects/Real-Time-Photon-Mapper/obj_parser/triangle.dat", "w");
    if (outfile == NULL)
    {
        fprintf(stderr, "\nError opend file\n");
        exit (1);
    }

    for (int i = 0; i < triangles->size(); ++i) {
        Binary_Triangle binary_triangle = construct_binary_triangle(triangles->at(i));
        fwrite(&(binary_triangle), sizeof(Binary_Triangle), 1, outfile);
    }

    if(fwrite != 0)
        printf("contents to file written successfully !\n");
    else
        printf("error writing file !\n");

    // close file
    fclose (outfile);
}

void triangle_read(std::vector<Triangle> *triangles) {
    FILE *infile;

    Binary_Triangle input;

    // Open person.dat for reading
    infile = fopen ("/Users/beaucarlborg/CLionProjects/Real-Time-Photon-Mapper/obj_parser/triangle.dat", "r");
    if (infile == NULL)
    {
        fprintf(stderr, "\nError opening file\n");
        exit (1);
    }

    // read file contents till end of file
    while(fread(&input, sizeof(Binary_Triangle), 1, infile)) {
        Triangle triangle = deconstruct_binary_triangle(input);

        triangles->push_back(triangle);
    }

    // close file
    fclose (infile);
}