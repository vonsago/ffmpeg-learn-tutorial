nvcc colorhdr.cu -c
ar rcs libcolorhdr.a colorhdr.o
g++ -o imgfile imgfile.c libcolorhdr.a -L/usr/local/cuda/lib64 -lcudart