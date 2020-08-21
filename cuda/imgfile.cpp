//
// Created by von on 2020/8/21.
//
#include <stdlib.h>
#include <stdio.h>
#include "colorhdr.h"

int main(int argc, char ** argv) {
    FILE *file = NULL, *dst_file = NULL;
    char *data = NULL;
    float *dst_data =NULL;
    int file_length = 0;

    file = fopen(argv[1], "rb");
    dst_file = fopen("ret.flv", "wb");

    fseek(file, 0, SEEK_END);
    file_length = ftell(file);

    data = (char *)malloc((file_length + 1) * sizeof(char));
    dst_data = (float *)malloc((file_length + 1) * sizeof(float));
    rewind(file);
    fread(data, 1, ftell(file), file);
    data[ftell(file)] = '\0';
    fclose(file);

    int i = doitgpu((float*)(data), 1,1,1, dst_data);
    printf("%d\n", i);

    fwrite(dst_data, 1, file_length, dst_file);
    fclose(dst_file);

    return 0;
}