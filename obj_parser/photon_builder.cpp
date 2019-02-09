//
// Created by Beau Carlborg on 2019-02-09.
//

// C program for writing
// struct to file


#include "./photon_builder.h"


#include <stdio.h>
#include <stdlib.h>
#include <string.h>


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
        fwrite(&(triangles->at(i)), sizeof(Triangle), 1, outfile);
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

    Triangle input;

    // Open person.dat for reading
    infile = fopen ("/Users/beaucarlborg/CLionProjects/Real-Time-Photon-Mapper/obj_parser/triangle.dat", "r");
    if (infile == NULL)
    {
        fprintf(stderr, "\nError opening file\n");
        exit (1);
    }

    // read file contents till end of file
    while(fread(&input, sizeof(struct Triangle), 1, infile)) {
        triangles->push_back(input);
    }

    // close file
    fclose (infile);
}






void person_write() {
    FILE *outfile;

    // open file for writing
    outfile = fopen ("/Users/beaucarlborg/CLionProjects/Real-Time-Photon-Mapper/obj_parser/person.dat", "w");
    if (outfile == NULL)
    {
        fprintf(stderr, "\nError opend file\n");
        exit (1);
    }

    struct person input1 = {1, "rohan", "sharma"};
    struct person input2 = {2, "mahendra", "dhoni"};

    // write struct to file
    fwrite (&input1, sizeof(struct person), 1, outfile);
    fwrite (&input2, sizeof(struct person), 1, outfile);

    if(fwrite != 0)
        printf("contents to file written successfully !\n");
    else
        printf("error writing file !\n");

    // close file
    fclose (outfile);
}

void person_read() {
    FILE *infile;
    struct person input;

    // Open person.dat for reading
    infile = fopen ("/Users/beaucarlborg/CLionProjects/Real-Time-Photon-Mapper/obj_parser/person.dat", "r");
    if (infile == NULL)
    {
        fprintf(stderr, "\nError opening file\n");
        exit (1);
    }

    // read file contents till end of file
    while(fread(&input, sizeof(struct person), 1, infile))
        printf ("id = %d name = %s %s\n", input.id,
                input.fname, input.lname);

    // close file
    fclose (infile);
}
