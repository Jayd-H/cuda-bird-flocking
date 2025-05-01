#include <helper_gl.h>
#include <GL/freeglut.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_functions.h>
#include <helper_cuda.h>

typedef unsigned int uint;
typedef unsigned char uchar;

#define REFRESH_DELAY 10  // ms
#define MAX_BIRDS 10000

StopWatchInterface* timer = 0;
uint width = 1024, height = 768;
dim3 blockSize(256);  // Each block handles 256 birds
dim3 gridSize((MAX_BIRDS + blockSize.x - 1) / blockSize.x);  // Calculate grid size based on bird count

GLuint pbo = 0;  // OpenGL pixel buffer object
struct cudaGraphicsResource* cuda_pbo_resource;  // handles OpenGL-CUDA exchange
GLuint displayTex = 0;

// Simulation parameters
int numBirds = 1000;
float minBounds[3] = { -50.0f, -50.0f, -50.0f };
float maxBounds[3] = { 50.0f, 50.0f, 50.0f };
float separationWeight = 3.0f;
float alignmentWeight = 1.0f;
float cohesionWeight = 0.5f;

void display();
void initGLBuffers();
void cleanup();
void keyboard(unsigned char key, int x, int y);
void reshape(int x, int y);
void timerEvent(int value);

// Function declarations for CUDA operations
extern "C" void initSimulation(int numBirds, float* minBounds, float* maxBounds);
extern "C" void updateSimulation(float dt, float separationWeight, float alignmentWeight, float cohesionWeight);
extern "C" void renderBirds(int width, int height, uchar4* output);
extern "C" void freeSimulation();

// Initialize OpenGL
void initGL(int* argc, char** argv) {
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(width, height);
    glutCreateWindow("Bird Flocking Simulation - CUDA Implementation");
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutReshapeFunc(reshape);
    glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
    glutCloseFunc(cleanup);

    if (!isGLVersionSupported(2, 0) || !areGLExtensionsSupported("GL_ARB_pixel_buffer_object")) {
        fprintf(stderr, "Required OpenGL extensions missing.");
        exit(EXIT_FAILURE);
    }
}

// Display results using OpenGL
void display() {
    static float lastTime = 0.0f;
    static int frameCount = 0;
    static float fps = 0.0f;

    frameCount++;
    float currentTime = sdkGetTimerValue(&timer) / 1000.0f;
    if (currentTime - lastTime > 1.0f) {
        fps = frameCount / (currentTime - lastTime);
        frameCount = 0;
        lastTime = currentTime;
        char title[256];
        sprintf(title, "Bird Flocking Simulation - CUDA - %d birds - FPS: %.1f", numBirds, fps);
        glutSetWindowTitle(title);
    }

    sdkStartTimer(&timer);

    // Update bird positions and velocities
    updateSimulation(0.016f, separationWeight, alignmentWeight, cohesionWeight);  // ~60fps

    // Map PBO to get CUDA device pointer
    uchar4* d_output;
    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&d_output, &num_bytes, cuda_pbo_resource));

    // Render birds to buffer
    renderBirds(width, height, d_output);

    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));

    // Display results
    glClear(GL_COLOR_BUFFER_BIT);

    // Download image from PBO to OpenGL texture
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glBindTexture(GL_TEXTURE_RECTANGLE_ARB, displayTex);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexSubImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, 0, 0, width, height, GL_BGRA, GL_UNSIGNED_BYTE, 0);
    glEnable(GL_TEXTURE_RECTANGLE_ARB);

    // Draw textured quad
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

    glDisable(GL_TEXTURE_RECTANGLE_ARB);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

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
    case 27:  // ESC key
        glutDestroyWindow(glutGetWindow());
        return;
    case '+':
        separationWeight *= 1.1f;
        printf("Separation weight: %.2f\n", separationWeight);
        break;
    case '-':
        separationWeight /= 1.1f;
        printf("Separation weight: %.2f\n", separationWeight);
        break;
    case '[':
        alignmentWeight *= 1.1f;
        printf("Alignment weight: %.2f\n", alignmentWeight);
        break;
    case ']':
        alignmentWeight /= 1.1f;
        printf("Alignment weight: %.2f\n", alignmentWeight);
        break;
    case ',':
        cohesionWeight *= 1.1f;
        printf("Cohesion weight: %.2f\n", cohesionWeight);
        break;
    case '.':
        cohesionWeight /= 1.1f;
        printf("Cohesion weight: %.2f\n", cohesionWeight);
        break;
    default:
        break;
    }
}

void reshape(int x, int y) {
    width = x;
    height = y;
    initGLBuffers();
    glViewport(0, 0, x, y);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
}

void cleanup() {
    freeSimulation();
    checkCudaErrors(cudaGraphicsUnregisterResource(cuda_pbo_resource));
    glDeleteBuffers(1, &pbo);
    glDeleteTextures(1, &displayTex);
    sdkDeleteTimer(&timer);
}

void initGLBuffers() {
    if (pbo) {
        checkCudaErrors(cudaGraphicsUnregisterResource(cuda_pbo_resource));
        glDeleteBuffers(1, &pbo);
    }

    // Create pixel buffer object for display
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, width * height * sizeof(uchar4), 0, GL_STREAM_DRAW_ARB);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard));

    // Create texture for display
    if (displayTex) {
        glDeleteTextures(1, &displayTex);
    }

    glGenTextures(1, &displayTex);
    glBindTexture(GL_TEXTURE_RECTANGLE_ARB, displayTex);
    glTexImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_RECTANGLE_ARB, 0);
}

int main(int argc, char** argv) {
    // Set the CUDA device
    int devID = findCudaDevice(argc, (const char**)argv);
    if (devID < 0) {
        printf("No CUDA Capable devices found, exiting...\n");
        exit(EXIT_FAILURE);
    }

    // Print the device info
    cudaDeviceProp deviceProps;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProps, devID));
    printf("CUDA device [%s] has %d Multi-Processors\n",
        deviceProps.name, deviceProps.multiProcessorCount);

    // Initialize GL
    initGL(&argc, argv);

    // Create the timer
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    // Initialize bird simulation
    initSimulation(numBirds, minBounds, maxBounds);

    // Initialize OpenGL buffers
    initGLBuffers();

    // Print controls
    printf("\nControls:\n");
    printf("ESC - Exit\n");
    printf("+ / - : Increase/decrease separation weight\n");
    printf("[ / ] : Increase/decrease alignment weight\n");
    printf(", / . : Increase/decrease cohesion weight\n\n");

    // Start rendering loop
    glutMainLoop();

    return 0;
}