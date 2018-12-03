#pragma once
#ifndef __HELPER_INCLUDES_H__
#define __HELPER_INCLUDES_H__

#include <fstream>
#include <iostream>
#include <math.h>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

// OpenGL Graphics includes
#include <GL/glew.h>
#if defined(__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#ifndef glutCloseFunc
#define glutCloseFunc glutWMCloseFunc
#endif
#else
#include <GL/freeglut.h>
#endif

// includes, cuda
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Utilities and timing functions
#include <helper_functions.h> // includes cuda.h and cuda_runtime_api.h

// CUDA helper functions
#include <helper_cuda.h>    // helper functions for CUDA error check
#include <helper_cuda_gl.h> // helper functions for CUDA/GL interop

#define G_CONSTANT 6.67408e-11

typedef struct {
  float4 gl_position;
  float4 color;
  float3 position;
  float3 speed;
  float3 acceleration;
  float mass;
} Vertex;

#endif