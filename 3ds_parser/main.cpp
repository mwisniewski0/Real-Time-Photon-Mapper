//
// Created by Beau Carlborg on 2019-01-28.
//

#include "main.h"
#include <stdio.h>



int main(int argc, char **argv) {
    Lib3dsFile *new_file = read_3ds_file(argv[1]);
    printf("\nThe file was parsed in a half sensible way!\nHere are the total number of meshes: %d\n", new_file->nmeshes);

    return 0;
}


Lib3dsFile* read_3ds_file(const char* file_path) {

    printf("***********FILE PATH FROM MAIN.CPP: %s", file_path);
    Lib3dsFile *parsed_file = lib3ds_file_open(file_path);

    if (!parsed_file) {
        printf("#####ERROR#####\n##Failed to open file %s##", file_path);
    }
    else {
        printf("successfully read file, there are %d meshes in the file\n", parsed_file->nmeshes);
    }

    return parsed_file;


}