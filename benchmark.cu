#include "main.h"
#include <chrono>

using namespace std;

void calculate_acceleration_cpu(Vertex *v, unsigned int n) {
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      float distance = sqrt(pow(v[i].position.x - v[j].position.x, 2) +
                            pow(v[i].position.y - v[j].position.y, 2) +
                            pow(v[i].position.z - v[j].position.z, 2));

      float magnitude = G_CONSTANT / pow(distance, 3);

      float3 vector;
      vector.x = magnitude * (v[i].position.x - v[j].position.x);
      vector.y = magnitude * (v[i].position.y - v[j].position.y);
      vector.z = magnitude * (v[i].position.z - v[j].position.z);

      v[i].acceleration.x += -vector.x * v[j].mass;
      v[i].acceleration.y += -vector.y * v[j].mass;
      v[i].acceleration.z += -vector.z * v[j].mass;

      v[j].acceleration.x += vector.x * v[i].mass;
      v[j].acceleration.y += vector.y * v[i].mass;
      v[j].acceleration.z += vector.z * v[i].mass;
    }
  }
}

void calculate_position_cpu(Vertex *v, unsigned int n, float delta) {
  for (int i = 0; i < n; i++) {
    v[i].speed.x += v[i].acceleration.x * delta;
    v[i].speed.y += v[i].acceleration.y * delta;
    v[i].speed.z += v[i].acceleration.z * delta;

    v[i].position.x += v[i].speed.x * delta;
    v[i].position.y += v[i].speed.y * delta;
    v[i].position.z += v[i].speed.z * delta;

    v[i].acceleration.x = 0.0f;
    v[i].acceleration.y = 0.0f;
    v[i].acceleration.z = 0.0f;
  }
}

__global__ void calculate_acceleration_gpu(Vertex *v, unsigned int n) {
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
    atomicAdd(&(v[i].acceleration.x), sub.x);
    atomicAdd(&(v[i].acceleration.y), sub.y);
    atomicAdd(&(v[i].acceleration.z), sub.z);
  }
}

__global__ void calculate_position_gpu(Vertex *v, unsigned int n, float delta) {
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

  int iterations = getCmdLineArgumentInt(argc, argv, "iterations=");
  if (!iterations) {
    cout << "Please specify the number of iterations with the option "
            "--iterations."
         << endl;
    exit(EXIT_FAILURE);
  }

  ifstream input;
  input.open(*filename);

  if (!input) {
    cout << "Problem opening file." << endl;
    exit(EXIT_FAILURE);
  }

  input >> delta;
  input >> n_vertices;

  Vertex *v = new Vertex[n_vertices];

  float mass, position_x, position_y, position_z, speed_x, speed_y, speed_z;
  for (int i = 0; i < n_vertices; i++) {
    input >> mass >> position_x >> position_y >> position_z >> speed_x >>
        speed_y >> speed_z;

    v[i].mass = mass;

    v[i].position.x = position_x;
    v[i].position.y = position_y;
    v[i].position.z = position_z;

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

  input.close();

  Vertex *d_v;
  Vertex *r_v = new Vertex[n_vertices];
  cudaMalloc(&d_v, sizeof(Vertex) * n_vertices);
  cudaMemcpy(d_v, v, sizeof(Vertex) * n_vertices, cudaMemcpyHostToDevice);

  // launchKernel (devPtr, DIM, dt);
  auto start_gpu = chrono::high_resolution_clock::now();

  int i = 0;
  while (i < iterations) {
    dim3 num_blocks_acceleration(n_vertices);
    dim3 num_threads_acceleration(n_vertices);
    calculate_acceleration_gpu<<<num_blocks_acceleration,
                                 num_threads_acceleration>>>(d_v, n_vertices);

    dim3 num_blocks_position(n_vertices);
    dim3 num_threads_position(n_vertices);
    calculate_position_gpu<<<num_blocks_position, num_threads_position>>>(
        d_v, n_vertices, delta);

    i++;
  }

  auto end_gpu = chrono::high_resolution_clock::now();

  cudaMemcpy(r_v, d_v, sizeof(Vertex) * n_vertices, cudaMemcpyDeviceToHost);
  cudaFree(d_v);

  auto start_cpu = chrono::high_resolution_clock::now();

  i = 0;
  while (i < iterations) {
    calculate_acceleration_cpu(v, n_vertices);
    calculate_position_cpu(v, n_vertices, delta);
    i++;
  }

  auto end_cpu = chrono::high_resolution_clock::now();

  int match = 1;
  for (i = 0; i < n_vertices; i++) {
    if (v[i].position.x - r_v[i].position.x >= 1) {
      cout << "Results DO NOT match" << endl;
      match = 0;
      break;
    }
  }

  if (match) {
    cout << "Results match" << endl;
  }

  chrono::duration<float, std::milli> duration_cpu = end_cpu - start_cpu;
  chrono::duration<float, std::milli> duration_gpu = end_gpu - start_gpu;

  cout << "CPU in milliseconds: " << duration_cpu.count()
       << ", GPU in milliseconds: " << duration_gpu.count()
       << ", speedup: " << (duration_cpu.count() / duration_gpu.count())
       << endl;

  delete[] v, r_v;
}