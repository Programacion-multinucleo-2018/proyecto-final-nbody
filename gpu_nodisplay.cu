#include "gpu.cuh"
#include "main.h"

using namespace std;

///////////////////////////////////////////////////////////////////////////////
//! Simple kernel to modify vertex positions in sine wave pattern
//! @param data  data in global memory
///////////////////////////////////////////////////////////////////////////////

__global__ void calculate_acceleration(Vertex *v, unsigned int n) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < n && j < n && i < j) {
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
  int n_vertices;
  float delta;

  char **filename = (char **)malloc(sizeof(char *));

  if (!getCmdLineArgumentString(argc, argv, "file=", filename)) {
    cout << "Please specify an input file with the option --file." << endl;
    exit(EXIT_FAILURE);
  }

  ifstream input(*filename);
  input >> delta;
  input >> n_vertices;

  Vertex *v = new Vertex[n_vertices];

  float mass, position_x, position_y, position_z, speed_x, speed_y, speed_z;
  for (int i = 0; i < n_vertices; i++) {
    input >> mass >> position_x >> position_y >> position_z >> speed_x >>
        speed_y >> speed_z;
    cout << mass << position_x << position_y << position_z << speed_x << speed_y
         << speed_z << endl;

    v[i].mass = mass;

    v[i].position.x = position_x;
    v[i].position.y = position_y;
    v[i].position.z = position_z;
    v[i].position.w = 1.0f;

    v[i].speed.x = speed_x;
    v[i].speed.y = speed_y;
    v[i].speed.z = speed_z;

    v[i].acceleration.x = 0.0f;
    v[i].acceleration.y = 0.0f;
    v[i].acceleration.z = 0.0f;

    float cr = (float)(rand() % 502) + 10.0f;
    float cg = (float)(rand() % 502) + 10.0f;
    float cb = (float)(rand() % 502) + 10.0f;
    v[i].color.x = cr / (float)512;
    v[i].color.y = cg / (float)512;
    v[i].color.z = cb / (float)512;
    v[i].color.w = 1.0f;
  }

  Vertex *d_v;
  cudaMalloc(&d_v, sizeof(Vertex) * n_vertices);
  cudaMemcpy(d_v, v, sizeof(Vertex) * n_vertices, cudaMemcpyHostToDevice);

  // launchKernel (devPtr, DIM, dt);
  dim3 numBlocks((int)ceil((float)n_vertices / 16.0),
                 (int)ceil((float)n_vertices / 16.0));
  dim3 numThreads(16, 16);
  calculate_acceleration<<<numBlocks, numThreads>>>(d_v, n_vertices);
  numBlocks.y = 1;
  numThreads.y = 1;
  calculate_position<<<numBlocks, numThreads>>>(d_v, n_vertices, delta);
  numBlocks.y = (int)ceil((float)n_vertices / 16.0);
  numThreads.y = 16;
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