#ifndef __BASIC_INTEROP_H__
#define __BASIC_INTEROP_H__

#include "../main.h"
#include "gpu.cuh"

///////////////////////////////////////////////////////////////////////////////
//! Simple kernel to modify vertex positions in sine wave pattern
//! @param data  data in global memory
///////////////////////////////////////////////////////////////////////////////
__global__ void kernel(Vertex *v, unsigned int n, float delta) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < n && j < n && i != j) {
    float distance = sqrt(pow(v[i].position.x - v[j].position.x, 2) +
                          pow(v[i].position.y - v[j].position.y, 2) +
                          pow(v[i].position.z - v[j].position.z, 2));
    float force = v[i].mass * v[j].mass / pow(distance, 3);
    float force_x = force * v[i].position.x;
    float force_y = force * v[i].position.y;
    float force_z = force * v[i].position.z;
    atomicAdd()
  }
}

void runCuda(cudaGraphicsResource **resource, Vertex *devPtr, int dim,
             float dt) {
  // Getting an actual address in device memory that can be passed to our
  // kernel. We achieve this by instructing the CUDA runtime to map the shared
  // resource and then by requesting a pointer to the mapped resource.
  checkCudaErrors(cudaGraphicsMapResources(1, resource, NULL));
  // devPtr is our device memory
  size_t size;
  checkCudaErrors(
      cudaGraphicsResourceGetMappedPointer((void **)&devPtr, &size, *resource));

  // launchKernel (devPtr, DIM, dt);
  dim3 numBlocks(dim / 16, dim / 16);
  dim3 numThreads(16, 16);
  kernel<<<numBlocks, numThreads>>>(devPtr, dim, dim, dt);

  // unmapping our shared resource. This call is important to make prior to
  // performing rendering tasks because it provides synchronization between the
  // CUDA and graphics portions of the application. Specifically, it implies
  // that all CUDA operations performed prior to the call to
  // cudaGraphicsUnmapResources() will complete before ensuing graphics calls
  // begin.
  checkCudaErrors(cudaGraphicsUnmapResources(1, resource, NULL));
}

void unregRes(cudaGraphicsResource **res) {
  checkCudaErrors(cudaGraphicsUnmapResources(1, res, NULL));
}

void chooseDev(int ARGC, const char **ARGV) { gpuGLDeviceInit(ARGC, ARGV); }

void regBuffer(cudaGraphicsResource **res, unsigned int &vbo) {
  // setting up graphics interoperability by notifying the CUDA runtime
  // that we intend to share the OpenGL buffer named vbo with CUDA.
  checkCudaErrors(
      cudaGraphicsGLRegisterBuffer(res, vbo, cudaGraphicsMapFlagsWriteDiscard));
}

#endif
