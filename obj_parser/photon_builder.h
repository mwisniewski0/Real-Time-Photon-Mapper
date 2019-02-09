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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// a struct to read and write
struct person
{
    int id;
    char fname[20];
    char lname[20];
};


void triangle_write(std::vector<Triangle> *triangles);
void triangle_read(std::vector<Triangle> *triangles);
void person_write();
void person_read();


#endif //PHOTON_PHOTON_BUILDER_H
