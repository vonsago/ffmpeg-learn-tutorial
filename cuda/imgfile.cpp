//
// Created by von on 2020/8/21.
//

#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include "colorhdr.h"

int main(int argc, char ** argv) {
    FILE *file = NULL, *dst_file = NULL;
    uint8_t *data = NULL;
    uint8_t *dst_data =NULL;
    int file_length = 0;

    file = fopen(argv[1], "rb");
    dst_file = fopen("ret.flv", "wb");

    fseek(file, 0, SEEK_END);
    file_length = ftell(file);

    data = (uint8_t *)malloc((file_length + 1) * sizeof(uint8_t));
    dst_data = (uint8_t *)malloc((file_length + 1) * sizeof(uint8_t));
    rewind(file);
    fread(data, 1, file_length, file);
    fclose(file);

    printf("length: %d\n", file_length);
    int i = doitgpu(data, 1,1,1, dst_data, file_length);

    for(int i=0;i<file_length;i++){
        if(data[i]!= dst_data[i])
            printf("%d %d\n", data[i], dst_data[i]);
    }

    fwrite(dst_data, 1, file_length, dst_file);
    fclose(dst_file);

    return 0;
}