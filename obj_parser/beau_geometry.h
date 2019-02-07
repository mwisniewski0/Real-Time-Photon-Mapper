//
// Created by Beau Carlborg on 2019-02-06.
//

#ifndef PHOTON_BEAU_GEOMETRY_H
#define PHOTON_BEAU_GEOMETRY_H

typedef struct Triangle {
    float v0[3];
    float v1[3];
    float v2[3];

    float v0vt[2];
    float v1vt[2];
    float v2vt[2];


} Triangle;


typedef struct Material {
    float diffuse[3]; // yes
    float specular[3];  // yes
    float transmittance[3];  // yes
    float shininess;  // yes
    int type;       // type(ish)
};



#endif //PHOTON_BEAU_GEOMETRY_H
