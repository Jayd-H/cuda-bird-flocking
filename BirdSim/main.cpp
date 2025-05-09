#include <helper_gl.h>
#include <GL/freeglut.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <algorithm>
#include <float.h>

typedef unsigned int uint;
typedef unsigned char uchar;

#define REFRESH_DELAY 10
#define MAX_BIRDS 5000
#define DEFAULT_BIRD_COUNT 1000
#define DEFAULT_BENCHMARK_STEPS 1000
#define DEFAULT_SCALING_STEPS 500
#define REPORT_INTERVAL_SECS 5
#define MAX_FPS_HISTORY 100

// Predefined flock sizes for scaling test
const int FLOCK_SIZES[] = { 100, 500, 1000, 5000 };
const int NUM_FLOCK_SIZES = sizeof(FLOCK_SIZES) / sizeof(FLOCK_SIZES[0]);

struct PerformanceMetrics {
    float forceCalculationTime;
    float positionUpdateTime;
    float renderTime;
    int stepsCompleted;
    float totalTime;
    float minFps;
    float maxFps;
    float avgFps;
};

StopWatchInterface* timer = 0;
uint width = 1024, height = 768;
dim3 blockSize(256);
dim3 gridSize((MAX_BIRDS + blockSize.x - 1) / blockSize.x);

// Visualization variables
GLuint pbo = 0;
struct cudaGraphicsResource* cuda_pbo_resource;
GLuint displayTex = 0;

// Performance tracking variables
float fpsHistory[MAX_FPS_HISTORY];
int fpsHistoryIndex = 0;
int fpsHistoryCount = 0;
float totalFrameTime = 0.0f;
float simulationTime = 0.0f;
float renderTime = 0.0f;
int fpsCounter = 0;
float lastReportTime = 0.0f;

// Simulation parameters
int numBirds = DEFAULT_BIRD_COUNT;
float minBounds[3] = { -50.0f, -50.0f, -50.0f };
float maxBounds[3] = { 50.0f, 50.0f, 50.0f };

float separationWeight = 1.0f;
float alignmentWeight = 1.0f;
float cohesionWeight = 1.0f;

// Function declarations
void display();
void initGLBuffers();
void cleanup();
void reshape(int x, int y);
void timerEvent(int value);
void runVisualization(int argc, char** argv);
void runBenchmarkMode(int birdCount, int steps);
void runScalingMode();
void printUsage();
void processExitKey(unsigned char key, int x, int y);
void printPerformanceReport();

// Function declarations for CUDA operations
extern "C" void initSimulation(int numBirds, float* minBounds, float* maxBounds);
extern "C" void updateSimulation(float dt, float separationWeight, float alignmentWeight, float cohesionWeight);
extern "C" void renderBirds(int width, int height, uchar4* output);
extern "C" void freeSimulation();
extern "C" void runBenchmark(int birdCount, int steps, float dt, float separationWeight, float alignmentWeight, float cohesionWeight);
extern "C" void runScalingTest(int* flockSizes, int numSizes, int steps, float dt, float separationWeight, float alignmentWeight, float cohesionWeight);
extern "C" void getPerformanceMetrics(PerformanceMetrics* metrics);

// Initialize OpenGL
void initGL(int* argc, char** argv) {
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(width, height);
    glutCreateWindow("Bird Flocking Simulation - CUDA Implementation");
    glutDisplayFunc(display);
    glutKeyboardFunc(processExitKey);
    glutReshapeFunc(reshape);
    glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
    glutCloseFunc(cleanup);

    if (!isGLVersionSupported(2, 0) || !areGLExtensionsSupported("GL_ARB_pixel_buffer_object")) {
        fprintf(stderr, "Required OpenGL extensions missing.");
        exit(EXIT_FAILURE);
    }
}

void printPerformanceReport() {
    if (fpsHistoryCount > 0) {
        float totalFps = 0.0f;
        float minFps = FLT_MAX;
        float maxFps = 0.0f;

        for (int i = 0; i < fpsHistoryCount; i++) {
            totalFps += fpsHistory[i];
            minFps = std::min(minFps, fpsHistory[i]);
            maxFps = std::max(maxFps, fpsHistory[i]);
        }

        float avgFps = totalFps / fpsHistoryCount;

        PerformanceMetrics cudaMetrics;
        getPerformanceMetrics(&cudaMetrics);

        printf("\n=== Final Performance Report ===\n");
        printf("Bird count: %d\n", numBirds);
        printf("Average FPS: %.1f\n", avgFps);
        printf("Min FPS: %.1f\n", minFps);
        printf("Max FPS: %.1f\n", maxFps);
        printf("Time breakdown per frame:\n");
        printf("  - Simulation: %.2fms (%.1f%%)\n",
            simulationTime * 1000.0f / fpsCounter,
            simulationTime * 100.0f / totalFrameTime);
        printf("  - Rendering: %.2fms (%.1f%%)\n",
            renderTime * 1000.0f / fpsCounter,
            renderTime * 100.0f / totalFrameTime);
        printf("Component breakdown in simulation:\n");
        printf("=== Performance Report ===\n");
        printf("Total steps: %d\n", cudaMetrics.stepsCompleted);
        printf("Total time: %.3f seconds\n", cudaMetrics.totalTime);
        printf("Steps per second: %.1f\n", cudaMetrics.stepsCompleted / cudaMetrics.totalTime);
        printf("Time breakdown:\n");
        printf("  - Force calculations: %.3fs (%.1f%%)\n",
            cudaMetrics.forceCalculationTime,
            (cudaMetrics.forceCalculationTime * 100.0f) / cudaMetrics.totalTime);
        printf("  - Position updates: %.3fs (%.1f%%)\n",
            cudaMetrics.positionUpdateTime,
            (cudaMetrics.positionUpdateTime * 100.0f) / cudaMetrics.totalTime);
        float overhead = cudaMetrics.totalTime -
            (cudaMetrics.forceCalculationTime + cudaMetrics.positionUpdateTime);
        printf("  - Other/overhead: %.3fs (%.1f%%)\n",
            overhead,
            (overhead * 100.0f) / cudaMetrics.totalTime);
        printf("=========================\n");
    }
}

// Display results using OpenGL
void display() {
    static float lastFrameTime = 0.0f;
    static StopWatchInterface* frameTimer = NULL;
    static StopWatchInterface* simTimer = NULL;
    static StopWatchInterface* renderTimer = NULL;

    if (frameTimer == NULL) {
        sdkCreateTimer(&frameTimer);
        sdkCreateTimer(&simTimer);
        sdkCreateTimer(&renderTimer);
        sdkStartTimer(&frameTimer);
        lastReportTime = sdkGetTimerValue(&frameTimer) / 1000.0f;
    }

    // Calculate elapsed time
    float currentTime = sdkGetTimerValue(&frameTimer) / 1000.0f;
    float dt = currentTime - lastFrameTime;
    lastFrameTime = currentTime;
    totalFrameTime += dt;

    fpsCounter++;

    // Update bird positions and velocities
    sdkStartTimer(&simTimer);
    updateSimulation(0.016f, separationWeight, alignmentWeight, cohesionWeight);
    sdkStopTimer(&simTimer);
    float simTime = sdkGetTimerValue(&simTimer) / 1000.0f;
    simulationTime += simTime;
    sdkResetTimer(&simTimer);

    // Map PBO to get CUDA device pointer
    sdkStartTimer(&renderTimer);
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
    glTexSubImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
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
    sdkStopTimer(&renderTimer);
    float rendTime = sdkGetTimerValue(&renderTimer) / 1000.0f;
    renderTime += rendTime;
    sdkResetTimer(&renderTimer);

    // Report FPS periodically
    if (currentTime - lastReportTime >= REPORT_INTERVAL_SECS) {
        float fps = fpsCounter / (currentTime - lastReportTime);

        // Store in history
        fpsHistory[fpsHistoryIndex] = fps;
        fpsHistoryIndex = (fpsHistoryIndex + 1) % MAX_FPS_HISTORY;
        if (fpsHistoryCount < MAX_FPS_HISTORY) {
            fpsHistoryCount++;
        }

        printf("FPS: %.1f | Birds: %d | Sim time: %.2fms | Render time: %.2fms\n",
            fps, numBirds,
            (simulationTime / fpsCounter) * 1000.0f,
            (renderTime / fpsCounter) * 1000.0f);

        char title[256];
        sprintf(title, "Bird Flocking Simulation - CUDA - %d birds - FPS: %.1f", numBirds, fps);
        glutSetWindowTitle(title);

        fpsCounter = 0;
        simulationTime = 0.0f;
        renderTime = 0.0f;
        lastReportTime = currentTime;
    }
}

// GLUT timer callback
void timerEvent(int value) {
    if (glutGetWindow()) {
        glutPostRedisplay();
        glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
    }
}

void processExitKey(unsigned char key, int /*x*/, int /*y*/) {
    if (key == 27) {  // ESC key
        printPerformanceReport();
        glutDestroyWindow(glutGetWindow());
    }
}

// Handle window resize
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

// Clean up resources
void cleanup() {
    freeSimulation();

    if (pbo) {
        cudaGraphicsUnregisterResource(cuda_pbo_resource);
        glDeleteBuffers(1, &pbo);
    }

    if (displayTex) {
        glDeleteTextures(1, &displayTex);
    }

    if (timer) {
        sdkDeleteTimer(&timer);
    }
}

// Initialize OpenGL buffers
void initGLBuffers() {
    if (pbo) {
        checkCudaErrors(cudaGraphicsUnregisterResource(cuda_pbo_resource));
        glDeleteBuffers(1, &pbo);
    }

    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, width * height * sizeof(uchar4), 0, GL_STREAM_DRAW_ARB);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard));

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

// Print program usage
void printUsage() {
    printf("Bird Flocking Simulator - CUDA Implementation\n\n");
    printf("Usage: \n");
    printf("  BirdSim                             Run with visualization\n");
    printf("  BirdSim benchmark                   Default: %d birds, %d steps\n", DEFAULT_BIRD_COUNT, DEFAULT_BENCHMARK_STEPS);
    printf("  BirdSim benchmark <birds>           <birds> birds, %d steps\n", DEFAULT_BENCHMARK_STEPS);
    printf("  BirdSim benchmark <birds> <steps>   <birds> birds, <steps> steps\n");
    printf("  BirdSim scaling                     Run benchmarks across different flock sizes\n");
    printf("\nWhen running the visual simulation, bird colour indicates strongest force:\n");
    printf("  Red = Separation\n");
    printf("  Green = Alignment\n");
    printf("  Blue = Cohesion\n\n");
    printf("Visual Controls:\n");
    printf("  ESC - Exit\n");
    printf("Weights (fixed):\n");
    printf("  Separation: %.1f\n", separationWeight);
    printf("  Alignment: %.1f\n", alignmentWeight);
    printf("  Cohesion: %.1f\n", cohesionWeight);
}

// Run visualization mode
void runVisualization(int argc, char** argv) {
    printf("Starting visualization with %d birds...\n", numBirds);

    // Initialize CUDA simulation
    initSimulation(numBirds, minBounds, maxBounds);

    // Initialize OpenGL
    initGL(&argc, argv);

    // Create the timer
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    // Initialize OpenGL buffers
    initGLBuffers();

    // Print controls
    printf("\nControls:\n");
    printf("ESC - Exit\n");
    printf("\nFixed weights:\n");
    printf("Separation: %.1f\n", separationWeight);
    printf("Alignment: %.1f\n", alignmentWeight);
    printf("Cohesion: %.1f\n\n", cohesionWeight);

    // Start rendering loop
    glutMainLoop();
}

// Run benchmark mode
void runBenchmarkMode(int birdCount, int steps) {
    printf("Running performance benchmark with %d birds for %d steps\n",
        birdCount, steps);

    // Initialize simulation
    initSimulation(birdCount, minBounds, maxBounds);

    // Run benchmark
    runBenchmark(birdCount, steps, 0.016f, separationWeight, alignmentWeight, cohesionWeight);

    // Clean up
    freeSimulation();
}

// Run scaling test mode
void runScalingMode() {
    printf("Running scaling test with various flock sizes\n");

    runScalingTest((int*)FLOCK_SIZES, NUM_FLOCK_SIZES, DEFAULT_SCALING_STEPS, 0.016f,
        separationWeight, alignmentWeight, cohesionWeight);
}

// Entry point
int main(int argc, char** argv) {
    int devID = findCudaDevice(argc, (const char**)argv);
    if (devID < 0) {
        printf("No CUDA Capable devices found, exiting...\n");
        exit(EXIT_FAILURE);
    }

    // Print device info
    cudaDeviceProp deviceProps;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProps, devID));
    printf("CUDA device [%s] has %d Multi-Processors\n",
        deviceProps.name, deviceProps.multiProcessorCount);

    // Check for mode-specific command line arguments
    if (argc > 1) {
        if (strcmp(argv[1], "benchmark") == 0) {
            int birdCount = DEFAULT_BIRD_COUNT;
            int steps = DEFAULT_BENCHMARK_STEPS;

            // Parse bird count
            if (argc > 2 && argv[2][0] != '-') {
                birdCount = atoi(argv[2]);
                if (birdCount <= 0) birdCount = DEFAULT_BIRD_COUNT;
                if (birdCount > MAX_BIRDS) birdCount = MAX_BIRDS;

                // Parse steps
                if (argc > 3 && argv[3][0] != '-') {
                    steps = atoi(argv[3]);
                    if (steps <= 0) steps = DEFAULT_BENCHMARK_STEPS;
                }
            }

            runBenchmarkMode(birdCount, steps);
            return 0;
        }
        else if (strcmp(argv[1], "scaling") == 0) {
            runScalingMode();
            return 0;
        }
        else if (strcmp(argv[1], "help") == 0 || strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0) {
            printUsage();
            return 0;
        }
    }

    // Default to visualization mode if no specific mode is given
    runVisualization(argc, argv);

    return 0;
}