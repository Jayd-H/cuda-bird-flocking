/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef BICUBICTEXTURE_CU_
#define BICUBICTEXTURE_CU_
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <helper_math.h>
 // includes, cuda
#include <helper_cuda.h>

#include "vec3.h"
#include "ray.h"
#include "hitable.h"
#include "hitable_list.h"
#include "sphere.h"
#include <float.h> // For FLT_MAX

typedef unsigned int uint;
typedef unsigned char uchar;

#define BALL_COUNT 1000
#define TOTAL_OBJECTS (BALL_COUNT + 5)

// these are vectors to store the positions, velocities, and colors of the balls
__device__ static vec3 ball_positions[BALL_COUNT];
__device__ static vec3 ball_velocities[BALL_COUNT];
__device__ static vec3 ball_colors[BALL_COUNT];
__device__ static bool initialised = false;


__device__ static int ticks = 1;

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}


// this is a bit of a hacky way to do colour dynamically
// ideally we would have some kind of nice material system in play
__device__ vec3 castRay(const ray& r, hitable** world) {
    hit_record rec;
    if ((*world)->hit(r, 0.0, FLT_MAX, rec)) {
        // calculate the center of the hit object using the formula from sphere.h
        // from normal = (p - center) / radius, we get: center = p - normal * radius
        vec3 calculated_center = rec.p - rec.normal * 0.2f;  // 0.2f is the radius of all balls

        // check if this center matches any of our balls
        for (int i = 0; i < BALL_COUNT; i++) {
            // calculate distance from calculated center to actual ball center
            float dist = (calculated_center - ball_positions[i]).length();

            // if they're very close, this is our ball
            if (dist < 0.01f) {  // small epsilon for floating-point comparison
                return ball_colors[i];  // return the stored color for this ball
            }
        }

        // For walls, use the normal-based coloring
        return 0.5f * vec3(rec.normal.x() + 1.0f, rec.normal.y() + 1.0f, rec.normal.z() + 1.0f);
    }
    else {
        vec3 unit_direction = unit_vector(r.direction());
        float t = 0.5f * (unit_direction.y() + 1.0f);
        return (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
    }
}

//* creates a list of hittables in a pretty little neon world
__global__ void create_world(hitable** d_list, hitable** d_world) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        if (!initialised) {
            for (int i = 0; i < BALL_COUNT; i++) {
				// each ball gets different initial positions, velocities, and colours
                float angle = 2.0f * 3.14159f * (float)i / BALL_COUNT;
                ball_positions[i] = vec3(0.7f * cos(angle), 0.7f * sin(angle), -1);
                ball_velocities[i] = vec3(0.02f * cos(angle + 0.5f), 0.02f * sin(angle + 0.5f), 0);

                ball_colors[i] = vec3(
                    0.5f + 0.5f * cos(angle),
                    0.5f + 0.5f * sin(angle),
                    0.5f + 0.5f * sin(angle + 1.0f)
                );
            }
            initialised = true;
        }

        // for loop that iterates through ball counts to check for collisions and move them about
        for (int i = 0; i < BALL_COUNT; i++) {
            ball_positions[i] += ball_velocities[i];

            // left wall collision
            if (ball_positions[i].x() < -2.0) {
                ball_velocities[i] = vec3(-ball_velocities[i].x(), ball_velocities[i].y(), ball_velocities[i].z());
				// change colour on collision
                ball_colors[i] = vec3(1.0, 0.2, 0.2);
            }
            // right wall collision
            if (ball_positions[i].x() > 2.0) {
                ball_velocities[i] = vec3(-ball_velocities[i].x(), ball_velocities[i].y(), ball_velocities[i].z());
                ball_colors[i] = vec3(0.2, 1.0, 0.2);
            }
            // bottom wall collision
            if (ball_positions[i].y() < -2.0) {
                ball_velocities[i] = vec3(ball_velocities[i].x(), -ball_velocities[i].y(), ball_velocities[i].z());
                ball_colors[i] = vec3(0.8, 0.8, 0.2);
            }
            // top wall collision
            if (ball_positions[i].y() > 2.0) {
                ball_velocities[i] = vec3(ball_velocities[i].x(), -ball_velocities[i].y(), ball_velocities[i].z());
                ball_colors[i] = vec3(0.2, 0.2, 1.0);
            }
            // back wall collision (this is not really needed but is good to be here for completedness sake)
            if (ball_positions[i].z() < -3.0) {
                ball_velocities[i] = vec3(ball_velocities[i].x(), ball_velocities[i].y(), -ball_velocities[i].z());
                ball_colors[i] = vec3(0.8, 0.2, 0.8);
            }
        }

        // creates all the balls
        for (int i = 0; i < BALL_COUNT; i++) {
            *(d_list + i) = new sphere(ball_positions[i], 0.2);
        }

        // creates five walls
        *(d_list + BALL_COUNT) = new sphere(vec3(-10002.0, 0, -3), 10000);      // left
        *(d_list + BALL_COUNT + 1) = new sphere(vec3(10002.0, 0, -1), 10000);   // right
        *(d_list + BALL_COUNT + 2) = new sphere(vec3(0, -10002.0, -1), 10000);  // bottom
        *(d_list + BALL_COUNT + 3) = new sphere(vec3(0, 10002.0, -1), 10000);   // top
        *(d_list + BALL_COUNT + 4) = new sphere(vec3(0, 0, -10003.0), 10000);   // back

        *d_world = new hitable_list(d_list, TOTAL_OBJECTS);
    }
}


__global__ void free_world(hitable** d_list, hitable** d_world) {
    for (int i = 0; i < TOTAL_OBJECTS; i++) {
        delete* (d_list + i);
    }
    delete* d_world;
}


cudaArray* d_imageArray = 0;

__global__ void d_render(uchar4* d_output, uint width, uint height, hitable** d_world) {
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;
    uint i = y * width + x;
    float u = x / (float)width; //----> [0, 1]x[0, 1]
    float v = y / (float)height;
    u = 2.0 * u - 1.0; //---> [-1, 1]x[-1, 1]
    v = -(2.0 * v - 1.0);
    u *= width / (float)height;
    u *= 2.0;
    v *= 2.0;
    vec3 eye = vec3(0, 0.5, 1.5);
    float distFrEye2Img = 1.0;;
    if ((x < width) && (y < height))
    {
        //for each pixel
        vec3 pixelPos = vec3(u, v, eye.z() - distFrEye2Img);
        //fire a ray:
        ray r;
        r.O = eye;
        r.Dir = pixelPos - eye; //view direction along negtive z-axis!
        vec3 col = castRay(r, d_world);
        float red = col.x();
        float green = col.y();
        float blue = col.z();
        d_output[i] = make_uchar4(red * 255, green * 255, blue * 255, 0);
    }
}

extern "C" void freeTexture() {
    checkCudaErrors(cudaFreeArray(d_imageArray));
}

extern "C" void render(int width, int height, dim3 blockSize, dim3 gridSize, uchar4* output) {
    // make our world of hitables
    hitable** d_list;
    checkCudaErrors(cudaMalloc((void**)&d_list, TOTAL_OBJECTS * sizeof(hitable*)));
    hitable** d_world;
    checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hitable*)));
    create_world << <1, 1 >> > (d_list, d_world);
    checkCudaErrors(cudaGetLastError());

    checkCudaErrors(cudaDeviceSynchronize());
    d_render << <gridSize, blockSize >> > (output, width, height, d_world);
    getLastCudaError("kernel failed");

    // Clean up
    free_world << <1, 1 >> > (d_list, d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_world));
}
#endif