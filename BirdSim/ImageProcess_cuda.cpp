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

 
 // OpenGL Graphics includes
#include <helper_gl.h>
#include <GL/freeglut.h>

// Includes
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// CUDA system and GL includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Helper functions
#include <helper_functions.h>  // CUDA SDK Helper functions
#include <helper_cuda.h>       // CUDA device initialization helper functions

typedef unsigned int uint;
typedef unsigned char uchar;


#ifndef MAX
#define MAX(a, b) ((a < b) ? b : a)
#endif

StopWatchInterface* timer = 0;
bool g_Verify = false;

int* pArgc = NULL;
char** pArgv = NULL;

#define REFRESH_DELAY 10  // ms


const char* srcImageFilename = "lena_bw.pgm";


uint width = 512, height = 512;
uint imageWidth, imageHeight;
dim3 blockSize(16, 16);
dim3 gridSize(width / blockSize.x, height / blockSize.y);


GLuint pbo = 0;                                  // OpenGL pixel buffer object
struct cudaGraphicsResource* cuda_pbo_resource;  // handles OpenGL-CUDA exchange
GLuint displayTex = 0;
GLuint bufferTex = 0;


float tx = -27.75f, ty = -189.0f;  // image translation
float scale = 0.125f;   // image scale
float cx, cy;                 // image centre

void display();
void initGLBuffers();

void cleanup();

#define GL_TEXTURE_TYPE GL_TEXTURE_RECTANGLE_ARB
//#define GL_TEXTURE_TYPE GL_TEXTURE_2D

extern "C" void initGL(int* argc, char** argv);
extern "C" void loadImageData(int argc, char** argv);

extern "C" void initTexture(int imageWidth, int imageHeight, uchar * h_data);
extern "C" void freeTexture();
extern "C" void render(int width, int height,  dim3 blockSize, dim3 gridSize,  uchar4 * output);


// display results using OpenGL (called by GLUT)
void display() {
    sdkStartTimer(&timer);

    // map PBO to get CUDA device pointer
    uchar4* d_output;
    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer(
        (void**)&d_output, &num_bytes, cuda_pbo_resource));
    render(imageWidth, imageHeight, blockSize, gridSize,
         d_output);

    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));

    // Common display path
    {
        // display results
        glClear(GL_COLOR_BUFFER_BIT);


        // download image from PBO to OpenGL texture
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
        glBindTexture(GL_TEXTURE_TYPE, displayTex);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexSubImage2D(GL_TEXTURE_TYPE, 0, 0, 0, width, height, GL_BGRA,
            GL_UNSIGNED_BYTE, 0);
        glEnable(GL_TEXTURE_TYPE);


        // draw textured quad
        glDisable(GL_DEPTH_TEST);
        glBegin(GL_QUADS);
        glTexCoord2f(0.0f, (GLfloat)height);
        glVertex2f(0.0f, 0.0f);
        glTexCoord2f((GLfloat)width, (GLfloat)height);
        glVertex2f(1.0f, 0.0f);
        glTexCoord2f((GLfloat)width, 0.0f);
        glVertex2f(1.0f, 1.0f);
        glTexCoord2f(0.0f, 0.0f);
        glVertex2f(0.0f, 1.0f);
        glEnd();
        glDisable(GL_TEXTURE_TYPE);
        glDisable(GL_FRAGMENT_PROGRAM_ARB);

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    }

    glutSwapBuffers();
    glutReportErrors();

    sdkStopTimer(&timer);

   
}

// GLUT callback functions
void timerEvent(int value) {
    if (glutGetWindow()) {
        glutPostRedisplay();
        glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
    }
}

void keyboard(unsigned char key, int /*x*/, int /*y*/) {
    switch (key) {
    case 27:
#if defined(__APPLE__) || defined(MACOSX)
        exit(EXIT_SUCCESS);
#else
        glutDestroyWindow(glutGetWindow());
        return;
#endif



    case '=':
    case '+':
        scale *= 0.5f;
        break;

    case '-':
        scale *= 2.0f;
        break;

    case 'r':
        scale = 1.0f;
        tx = ty = 0.0f;
        break;

    case 'd':
        printf("%f, %f, %f\n", tx, ty, scale);
        break;


    default:
        break;
    }

}

int ox, oy;
int buttonState = 0;


void reshape(int x, int y) {
    width = x;
    height = y;
    imageWidth = width;
    imageHeight = height;

    initGLBuffers();

    glViewport(0, 0, x, y);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
}

void cleanup() {
    freeTexture();
    checkCudaErrors(cudaGraphicsUnregisterResource(cuda_pbo_resource));

    glDeleteBuffers(1, &pbo);


    glDeleteTextures(1, &displayTex);

    sdkDeleteTimer(&timer);
}

int iDivUp(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }

void initGLBuffers() {
    if (pbo) {
        // delete old buffer
        checkCudaErrors(cudaGraphicsUnregisterResource(cuda_pbo_resource));
        glDeleteBuffers(1, &pbo);
    }

    // create pixel buffer object for display
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, width * height * sizeof(uchar4), 0,
        GL_STREAM_DRAW_ARB);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    checkCudaErrors(cudaGraphicsGLRegisterBuffer(
        &cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard));


    // create texture for display
    if (displayTex) {
        glDeleteTextures(1, &displayTex);
    }

    glGenTextures(1, &displayTex);
    glBindTexture(GL_TEXTURE_TYPE, displayTex);
    glTexImage2D(GL_TEXTURE_TYPE, 0, GL_RGBA8, width, height, 0, GL_RGBA,
        GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_TYPE, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_TYPE, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_TYPE, 0);


    // calculate new grid size
    gridSize = dim3(iDivUp(width, blockSize.x), iDivUp(height, blockSize.y));
}

void mainMenu(int i) { keyboard(i, 0, 0); }

void initMenus() {
    glutCreateMenu(mainMenu);
    glutAddMenuEntry("Zoom in      [=]", '=');
    glutAddMenuEntry("Zoom out     [-]", '-');
    glutAddMenuEntry("Quit       [esc]", 27);
    glutAttachMenu(GLUT_RIGHT_BUTTON);
}


GLuint compileASMShader(GLenum program_type, const char* code) {
    GLuint program_id;
    glGenProgramsARB(1, &program_id);
    glBindProgramARB(program_type, program_id);
    glProgramStringARB(program_type, GL_PROGRAM_FORMAT_ASCII_ARB,
        (GLsizei)strlen(code), (GLubyte*)code);

    GLint error_pos;
    glGetIntegerv(GL_PROGRAM_ERROR_POSITION_ARB, &error_pos);

    if (error_pos != -1) {
        const GLubyte* error_string;
        error_string = glGetString(GL_PROGRAM_ERROR_STRING_ARB);
        fprintf(stderr, "Program error at position: %d\n%s\n", (int)error_pos,
            error_string);
        return 0;
    }

    return program_id;
}

void initialize(int argc, char** argv) {
    
    initGL(&argc, argv);

    // use command-line specified CUDA device, otherwise use device with highest
    // Gflops/s
    int devID = findCudaDevice(argc, (const char**)argv);

    // get number of SMs on this GPU
    cudaDeviceProp deviceProps;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProps, devID));
    printf("CUDA device [%s] has %d Multi-Processors\n", deviceProps.name,
        deviceProps.multiProcessorCount);

    // Create the timer (for fps measurement)
    sdkCreateTimer(&timer);

    // load image from disk
    loadImageData(argc, argv);

    printf(
        "\n"
        "\tControls\n"
        "\t=/- : Zoom in/out\n"
        "\t[esc] - Quit\n\n"               
        );

    initGLBuffers();

}

void initGL(int* argc, char** argv) {
    // initialize GLUT callback functions
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_ALPHA | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowSize(width, height);
    glutCreateWindow("CUDA bicubic texture filtering");
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
   
    glutReshapeFunc(reshape);
    glutTimerFunc(REFRESH_DELAY, timerEvent, 0);

    glutCloseFunc(cleanup);

    initMenus();

    if (!isGLVersionSupported(2, 0) ||
        !areGLExtensionsSupported("GL_ARB_pixel_buffer_object")) {
        fprintf(stderr, "Required OpenGL extensions are missing.");
        exit(EXIT_FAILURE);
    }

}

void loadImageData(int argc, char** argv) {
    // load image from disk
    uchar* h_data = NULL;
    char* srcImagePath = NULL;

    if ((srcImagePath = sdkFindFilePath(srcImageFilename, argv[0])) == NULL) {
        printf("bicubicTexture loadImageData() could not find <%s>\nExiting...\n",
            srcImageFilename);
        exit(EXIT_FAILURE);
    }

    sdkLoadPGM<unsigned char>(srcImagePath, &h_data, &imageWidth, &imageHeight);

    printf("Loaded '%s', %d x %d pixels\n", srcImageFilename, imageWidth,
        imageHeight);

    cx = imageWidth * 0.5f;
    cy = imageHeight * 0.5f;

    // initialize texture
    initTexture(imageWidth, imageHeight, h_data);
}


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {
    pArgc = &argc;
    pArgv = argv;

    // parse arguments
    char* filename;

#if defined(__linux__)
    setenv("DISPLAY", ":0", 0);
#endif

    printf("Starting Original Texture\n");

        // This runs the CUDA kernel (bicubicFiltering) + OpenGL visualization
        initialize(argc, argv);
        glutMainLoop();
  
    exit(EXIT_SUCCESS);
}