#include "gpu.cuh"
#include "main.h"

///////////////////////////////////////////////////////////////////////////////
//! Simple kernel to modify vertex positions in sine wave pattern
//! @param data  data in global memory
///////////////////////////////////////////////////////////////////////////////

__global__ void calculate_acceleration(Vertex *v, unsigned int n) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < n && j < n && i != j) {
    float distance = sqrt(pow(v[i].position.x - v[j].position.x, 2) +
                          pow(v[i].position.y - v[j].position.y, 2) +
                          pow(v[i].position.z - v[j].position.z, 2));

    float magnitude = G_CONSTANT / pow(distance, 3);

    float3 vector;
    vector.x = magnitude * (v[i].position.x - v[j].position.x);
    vector.y = magnitude * (v[i].position.y - v[j].position.y);
    vector.z = magnitude * (v[i].position.z - v[j].position.z);

    atomicAdd(&(v[i].acceleration.x), -vector.x * v[j].mass);
    atomicAdd(&(v[i].acceleration.y), -vector.y * v[j].mass);
    atomicAdd(&(v[i].acceleration.z), -vector.z * v[j].mass);

    atomicAdd(&(v[j].acceleration.x), vector.x * v[i].mass);
    atomicAdd(&(v[j].acceleration.y), vector.y * v[i].mass);
    atomicAdd(&(v[j].acceleration.z), vector.z * v[i].mass);
  }
}

__global__ void calculate_position(Vertex *v, unsigned int n, float delta) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n) {
    v[i].speed.x += v[i].acceleration.x * delta;
    v[i].speed.y += v[i].acceleration.y * delta;
    v[i].speed.z += v[i].acceleration.z * delta;

    v[i].position.x += v[i].speed.x * delta;
    v[i].position.y += v[i].speed.y * delta;
    v[i].position.z += v[i].speed.z * delta;

    v[i].acceleration.x = 0.0f;
    v[i].acceleration.y = 0.0f;
    v[i].acceleration.z = 0.0f;

    printf("#%i\nSpeed: %f %f %f\nPosition: %f %f %f\n", i, v[i].speed.x,
           v[i].speed.y, v[i].speed.z, v[i].position.x, v[i].position.y,
           v[i].position.z);
  }
}

int main(int argc, const char **argv) {
  int n_vertices = 2;
  float delta = 1.0f;
  Vertex *v = new Vertex[2];

  v[0].mass = 5.972e+24;

  v[0].position.x = 0.0f;
  v[0].position.y = 0.0f;
  v[0].position.z = 0.0f;
  v[0].position.w = 1.0f;

  v[0].speed.x = 0.0f;
  v[0].speed.y = 0.0f;
  v[0].speed.z = 0.0f;

  v[0].acceleration.x = 0.0f;
  v[0].acceleration.y = 0.0f;
  v[0].acceleration.z = 0.0f;

  v[1].mass = 1;

  v[1].position.x = 6000000;
  v[1].position.y = 0.0f;
  v[1].position.z = 0.0f;
  v[1].position.w = 1.0f;

  v[1].speed.x = 0.0f;
  v[1].speed.y = 0.0f;
  v[1].speed.z = 0.0f;

  v[1].acceleration.x = 0.0f;
  v[1].acceleration.y = 0.0f;
  v[1].acceleration.z = 0.0f;

  Vertex *d_v;
  cudaMalloc(&d_v, sizeof(Vertex) * 2);
  cudaMemcpy(d_v, v, sizeof(Vertex) * 2, cudaMemcpyHostToDevice);

  // launchKernel (devPtr, DIM, dt);
  dim3 numBlocks((int)ceil((float)n_vertices / 16.0),
                 (int)ceil((float)n_vertices / 16.0));
  dim3 numThreads(16, 16);
  calculate_acceleration<<<numBlocks, numThreads>>>(d_v, n_vertices);
  numBlocks.y = 1;
  numThreads.y = 1;
  calculate_position<<<numBlocks, numThreads>>>(d_v, n_vertices, delta);

  // unmapping our shared resource. This call is important to make prior to
  // performing rendering tasks because it provides synchronization between the
  // CUDA and graphics portions of the application. Specifically, it implies
  // that all CUDA operations performed prior to the call to
  // cudaGraphicsUnmapResources() will complete before ensuing graphics calls
  // begin.
  cudaFree(d_v);
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