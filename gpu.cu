#ifndef __BASIC_INTEROP_H__
#define __BASIC_INTEROP_H__

#include "gpu.cuh"
#include "main.h"

///////////////////////////////////////////////////////////////////////////////
//! Simple kernel to modify vertex positions in sine wave pattern
//! @param data  data in global memory
///////////////////////////////////////////////////////////////////////////////

__global__ void calculate_acceleration(Vertex *v, unsigned int n) {
  unsigned int i = blockIdx.x;
  unsigned int j = threadIdx.x;

  __shared__ float3 sub;

  if (j == 0) {
    sub.x = 0.0f;
    sub.y = 0.0f;
    sub.z = 0.0f;
  }

  __syncthreads();

  if (i < n && j < n && i != j) {
    float distance = sqrt(pow(v[i].position.x - v[j].position.x, 2) +
                          pow(v[i].position.y - v[j].position.y, 2) +
                          pow(v[i].position.z - v[j].position.z, 2));

    float magnitude = G_CONSTANT / pow(distance, 3);

    float3 vector;
    vector.x = magnitude * (v[i].position.x - v[j].position.x);
    vector.y = magnitude * (v[i].position.y - v[j].position.y);
    vector.z = magnitude * (v[i].position.z - v[j].position.z);

    atomicAdd(&(sub.x), -vector.x * v[j].mass);
    atomicAdd(&(sub.y), -vector.y * v[j].mass);
    atomicAdd(&(sub.z), -vector.z * v[j].mass);
  }

  __syncthreads();

  if (j == 0) {
    v[i].acceleration.x = sub.x;
    v[i].acceleration.y = sub.y;
    v[i].acceleration.z = sub.z;
  }
}

__global__ void calculate_position(Vertex *v, unsigned int n, float delta,
                                   float max_distance) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n) {
    v[i].speed.x += v[i].acceleration.x * delta;
    v[i].speed.y += v[i].acceleration.y * delta;
    v[i].speed.z += v[i].acceleration.z * delta;

    v[i].position.x += v[i].speed.x * delta;
    v[i].position.y += v[i].speed.y * delta;
    v[i].position.z += v[i].speed.z * delta;

    v[i].gl_position.x = v[i].position.x / max_distance;
    v[i].gl_position.y = v[i].position.y / max_distance;
    v[i].gl_position.z = v[i].position.z / max_distance;

    v[i].acceleration.x = 0.0f;
    v[i].acceleration.y = 0.0f;
    v[i].acceleration.z = 0.0f;
  }
}

void runCuda(cudaGraphicsResource **resource, Vertex *devPtr, int n_vertices,
             float delta, float max_distance, int *iteration) {
  (*iteration)++;

  checkCudaErrors(cudaGraphicsMapResources(1, resource, NULL));

  size_t size;
  checkCudaErrors(
      cudaGraphicsResourceGetMappedPointer((void **)&devPtr, &size, *resource));

  // launchKernel (devPtr, DIM, dt);
  dim3 num_blocks_acceleration(n_vertices);
  dim3 num_threads_acceleration(n_vertices);
  calculate_acceleration<<<num_blocks_acceleration, num_threads_acceleration>>>(
      devPtr, n_vertices);

  dim3 num_blocks_position(n_vertices);
  dim3 num_threads_position(n_vertices);
  calculate_position<<<num_blocks_position, num_threads_position>>>(
      devPtr, n_vertices, delta, max_distance);
  cudaDeviceSynchronize();

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
