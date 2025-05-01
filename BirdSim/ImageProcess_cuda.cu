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

#ifndef _BICUBICTEXTURE_CU_
#define _BICUBICTEXTURE_CU_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <helper_math.h>

 // includes, cuda
#include <helper_cuda.h>

typedef unsigned int uint;
typedef unsigned char uchar;


cudaArray* d_imageArray = 0;

cudaTextureObject_t rgbaTexdImage;


__global__ void d_render(uchar4* d_output, uint width, uint height, float tx,
    float ty, float scale, float cx, float cy,
    cudaTextureObject_t texObj) {

    // lecturer says no
    // uint x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    // uint y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    // uint i = __umul24(y, width) + x;

    // basically is the same before
	// the umul24() function is an optimisation for multiplaction of 24-bit integers
    // it is mainly for older GPUs i believe

    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;
    uint i = y * width + x;

    // translate image
    // float2 T = { 20, 10 };
    // float u = x + T.x;
    // float v = y + T.y;

    // scale image
	// float2 S = { 1.2, 0.5 };
	// float u = x * S.x;
	// float v = y * S.y;

    // rotate image
    // float angle = 0.5;
    // float u = x * cos(angle) - y * sin(angle);
    // float v = x * sin(angle) + y * cos(angle);

    // scale by position
    // float s = 1.5;
    // float centerx = width / 2; // center x
    // float centery = height / 2; // center y
    // float u = (x - centerx) * s + centerx;
    // float v = (y - centery) * s + centery;


    // roate by image center
    // float angle = 0.5;
    // float centerx = width / 2; 
    // float centery = height / 2;

    // float dx = x - centerx;
    // float dy = y - centery;

    // float u = cos(angle) * dx - sin(angle) * dy + centerx;
    // float v = sin(angle) * dx + cos(angle) * dy + centery;


    // scale around center and translate
    float tx = 20.0f;
    float ty = 10.0f;
    float s = 1.2f; 
    float centerx = width / 2; 
    float centery = height / 2; 

    float u = (x - centerx) * s + centerx + tx;
    float v = (y - centery) * s + centery + ty;


    // original lines without any sorcery
    // float u = (x - cx) * scale + cx + tx;
    // float v = (y - cy) * scale + cy + ty;

    if ((x < width) && (y < height)) {
        // write output color
        float c = tex2D<float>(texObj, u, v);

        // standard image
        d_output[i] = make_uchar4(c * 0xff, c * 0xff, c * 0xff, 0);

        // pure green
        // d_output[i] = make_uchar4(0, 0xff, 0, 0);

        // pure red
        // d_output[i] = make_uchar4(0, 0, 0xff, 0);

        // pure blue
		// d_output[i] = make_uchar4(0xff, 0, 0, 0);

    }
}


extern "C" void initTexture(int imageWidth, int imageHeight, uchar * h_data) {
    // allocate array and copy image data
    cudaChannelFormatDesc channelDesc =
        cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
    checkCudaErrors(
        cudaMallocArray(&d_imageArray, &channelDesc, imageWidth, imageHeight));
    checkCudaErrors(cudaMemcpy2DToArray(
        d_imageArray, 0, 0, h_data, imageWidth * sizeof(uchar),
        imageWidth * sizeof(uchar), imageHeight, cudaMemcpyHostToDevice));
    free(h_data);

    cudaResourceDesc texRes;
    memset(&texRes, 0, sizeof(cudaResourceDesc));

    texRes.resType = cudaResourceTypeArray;
    texRes.res.array.array = d_imageArray;

    cudaTextureDesc texDescr;
    memset(&texDescr, 0, sizeof(cudaTextureDesc));

    texDescr.normalizedCoords = false;
    texDescr.filterMode = cudaFilterModeLinear;
    texDescr.addressMode[0] = cudaAddressModeClamp;
    texDescr.addressMode[1] = cudaAddressModeClamp;
    texDescr.readMode = cudaReadModeNormalizedFloat;

    checkCudaErrors(
        cudaCreateTextureObject(&rgbaTexdImage, &texRes, &texDescr, NULL));

}

extern "C" void freeTexture() {

    checkCudaErrors(cudaFreeArray(d_imageArray));
}

// render image using CUDA
extern "C" void render(int width, int height,  dim3 blockSize, dim3 gridSize,
     uchar4 * output) {

    float tx = 0, ty = 0, scale = 1, cx = 0, cy = 0;

        d_render << <gridSize, blockSize >> > (output, width, height, 0, 0, 1,
            0, 0, rgbaTexdImage);


    getLastCudaError("kernel failed");
}

#endif