#ifndef __BASIC_INTEROP_H__
#define __BASIC_INTEROP_H__

#include "../main.h"

///////////////////////////////////////////////////////////////////////////////
//! Simple kernel to modify vertex positions in sine wave pattern
//! @param data  data in global memory
///////////////////////////////////////////////////////////////////////////////
__global__ void kernel(Vertex *v, unsigned int width, unsigned int height,
                       float time);

void runCuda(cudaGraphicsResource **resource, Vertex *devPtr, int dim,
             float dt);

void unregRes(cudaGraphicsResource **res);

void chooseDev(int ARGC, const char **ARGV);

void regBuffer(cudaGraphicsResource **res, unsigned int &vbo);

#endif
