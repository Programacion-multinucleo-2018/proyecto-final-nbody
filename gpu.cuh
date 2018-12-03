#ifndef __BASIC_INTEROP_H__
#define __BASIC_INTEROP_H__

#include "main.h"

void runCuda(cudaGraphicsResource **resource, Vertex *devPtr, int n_vertices,
             float delta, float max_distance, int *iteration);

void unregRes(cudaGraphicsResource **res);

void chooseDev(int ARGC, const char **ARGV);

void regBuffer(cudaGraphicsResource **res, unsigned int &vbo);

#endif
