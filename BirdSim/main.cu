#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <helper_math.h>
#include <helper_cuda.h>
#include <curand_kernel.h>
#include <time.h>

#define MAX_BIRDS 10000
#define EPSILON 1e-6f
#define PERCEPTION_RADIUS 15.0f
#define SEPARATION_RADIUS_FACTOR 0.3f
#define MAX_SPEED 3.0f
#define MAX_FORCE 0.1f

typedef unsigned char uchar;

struct PerformanceMetrics {
    float forceCalculationTime;
    float positionUpdateTime;
    int stepsCompleted;
    float totalTime;
};

struct Bird {
    float3 position;
    float3 velocity;
    float3 acceleration;
    int dominantForce;
};

__device__ __managed__ Bird birds[MAX_BIRDS];
__device__ __managed__ int numBirds;
__device__ __managed__ float3 minBounds;
__device__ __managed__ float3 maxBounds;
__device__ __managed__ PerformanceMetrics metrics;

curandState* d_states = nullptr;
cudaEvent_t startEvent, stopEvent, forceStartEvent, forceStopEvent, posStartEvent, posStopEvent;

__global__ void setupRNG(curandState* states) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < numBirds) {
        curand_init(clock64(), tid, 0, &states[tid]);
    }
}

__global__ void initBirds(curandState* states) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < numBirds) {
        curandState localState = states[tid];

        birds[tid].position.x = minBounds.x + curand_uniform(&localState) * (maxBounds.x - minBounds.x);
        birds[tid].position.y = minBounds.y + curand_uniform(&localState) * (maxBounds.y - minBounds.y);
        birds[tid].position.z = minBounds.z + curand_uniform(&localState) * (maxBounds.z - minBounds.z);

        float3 vel;
        vel.x = curand_uniform(&localState) * 2.0f - 1.0f;
        vel.y = curand_uniform(&localState) * 2.0f - 1.0f;
        vel.z = curand_uniform(&localState) * 2.0f - 1.0f;

        float len = sqrtf(vel.x * vel.x + vel.y * vel.y + vel.z * vel.z);
        if (len > EPSILON) {
            vel.x /= len;
            vel.y /= len;
            vel.z /= len;
        }

        float speed = 0.5f + curand_uniform(&localState) * 1.5f;
        birds[tid].velocity = make_float3(vel.x * speed, vel.y * speed, vel.z * speed);
        birds[tid].acceleration = make_float3(0.0f, 0.0f, 0.0f);
        birds[tid].dominantForce = 0;

        states[tid] = localState;
    }
}

__device__ float3 calculateWrappedDistance(float3 pos1, float3 pos2) {
    float3 diff = make_float3(pos1.x - pos2.x, pos1.y - pos2.y, pos1.z - pos2.z);

    float size_x = maxBounds.x - minBounds.x;
    float size_y = maxBounds.y - minBounds.y;
    float size_z = maxBounds.z - minBounds.z;

    if (abs(diff.x) > size_x * 0.5f) {
        diff.x = diff.x - copysignf(size_x, diff.x);
    }
    if (abs(diff.y) > size_y * 0.5f) {
        diff.y = diff.y - copysignf(size_y, diff.y);
    }
    if (abs(diff.z) > size_z * 0.5f) {
        diff.z = diff.z - copysignf(size_z, diff.z);
    }

    return diff;
}

__global__ void calculateForces(float separationWeight, float alignmentWeight, float cohesionWeight) {
    extern __shared__ Bird sharedBirds[];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < numBirds) {
        Bird& bird = birds[tid];

        float3 separation = make_float3(0.0f, 0.0f, 0.0f);
        float3 alignment = make_float3(0.0f, 0.0f, 0.0f);
        float3 cohesion = make_float3(0.0f, 0.0f, 0.0f);
        int separationCount = 0;
        int alignmentCount = 0;
        int cohesionCount = 0;

        for (int chunkStart = 0; chunkStart < numBirds; chunkStart += blockDim.x) {
            int chunkIdx = chunkStart + threadIdx.x;
            if (chunkIdx < numBirds) {
                sharedBirds[threadIdx.x] = birds[chunkIdx];
            }

            __syncthreads();

            int chunkSize = min(blockDim.x, numBirds - chunkStart);
            for (int j = 0; j < chunkSize; j++) {
                int otherIdx = chunkStart + j;
                if (tid == otherIdx) continue;

                Bird& other = sharedBirds[j];

                // Early rejection check - just compare one axis first
                float dx = abs(bird.position.x - other.position.x);
                float size_x = maxBounds.x - minBounds.x;
                if (dx > size_x * 0.5f) {
                    dx = size_x - dx;
                }
                if (dx > PERCEPTION_RADIUS) continue;

                float3 diff = calculateWrappedDistance(bird.position, other.position);
                float dist_sq = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;

                if (dist_sq > EPSILON && dist_sq < (PERCEPTION_RADIUS * SEPARATION_RADIUS_FACTOR) * (PERCEPTION_RADIUS * SEPARATION_RADIUS_FACTOR)) {
                    float3 repulse = make_float3(diff.x / dist_sq, diff.y / dist_sq, diff.z / dist_sq);
                    separation.x += repulse.x;
                    separation.y += repulse.y;
                    separation.z += repulse.z;
                    separationCount++;
                }

                if (dist_sq < PERCEPTION_RADIUS * PERCEPTION_RADIUS) {
                    alignment.x += other.velocity.x;
                    alignment.y += other.velocity.y;
                    alignment.z += other.velocity.z;
                    alignmentCount++;

                    float3 otherPos = make_float3(
                        bird.position.x - diff.x,
                        bird.position.y - diff.y,
                        bird.position.z - diff.z
                    );
                    cohesion.x += otherPos.x;
                    cohesion.y += otherPos.y;
                    cohesion.z += otherPos.z;
                    cohesionCount++;
                }
            }

            __syncthreads();
        }

        float3 separationForce = make_float3(0.0f, 0.0f, 0.0f);
        float3 alignmentForce = make_float3(0.0f, 0.0f, 0.0f);
        float3 cohesionForce = make_float3(0.0f, 0.0f, 0.0f);

        if (separationCount > 0) {
            separation.x /= separationCount;
            separation.y /= separationCount;
            separation.z /= separationCount;

            float len = sqrtf(separation.x * separation.x + separation.y * separation.y + separation.z * separation.z);
            if (len > EPSILON) {
                separation.x /= len;
                separation.y /= len;
                separation.z /= len;
            }

            separationForce = make_float3(
                separation.x * MAX_FORCE,
                separation.y * MAX_FORCE,
                separation.z * MAX_FORCE
            );
        }

        if (alignmentCount > 0) {
            alignment.x /= alignmentCount;
            alignment.y /= alignmentCount;
            alignment.z /= alignmentCount;

            float3 steer = make_float3(
                alignment.x - bird.velocity.x,
                alignment.y - bird.velocity.y,
                alignment.z - bird.velocity.z
            );

            float len = sqrtf(steer.x * steer.x + steer.y * steer.y + steer.z * steer.z);
            if (len > MAX_FORCE && len > EPSILON) {
                steer.x = (steer.x / len) * MAX_FORCE;
                steer.y = (steer.y / len) * MAX_FORCE;
                steer.z = (steer.z / len) * MAX_FORCE;
            }

            alignmentForce = steer;
        }

        if (cohesionCount > 0) {
            cohesion.x /= cohesionCount;
            cohesion.y /= cohesionCount;
            cohesion.z /= cohesionCount;

            float3 desired = make_float3(
                cohesion.x - bird.position.x,
                cohesion.y - bird.position.y,
                cohesion.z - bird.position.z
            );

            float len = sqrtf(desired.x * desired.x + desired.y * desired.y + desired.z * desired.z);
            if (len > EPSILON) {
                desired.x = (desired.x / len) * MAX_SPEED;
                desired.y = (desired.y / len) * MAX_SPEED;
                desired.z = (desired.z / len) * MAX_SPEED;
            }

            float3 steer = make_float3(
                desired.x - bird.velocity.x,
                desired.y - bird.velocity.y,
                desired.z - bird.velocity.z
            );

            len = sqrtf(steer.x * steer.x + steer.y * steer.y + steer.z * steer.z);
            if (len > MAX_FORCE && len > EPSILON) {
                steer.x = (steer.x / len) * MAX_FORCE;
                steer.y = (steer.y / len) * MAX_FORCE;
                steer.z = (steer.z / len) * MAX_FORCE;
            }

            cohesionForce = steer;
        }

        separationForce.x *= separationWeight;
        separationForce.y *= separationWeight;
        separationForce.z *= separationWeight;

        alignmentForce.x *= alignmentWeight;
        alignmentForce.y *= alignmentWeight;
        alignmentForce.z *= alignmentWeight;

        cohesionForce.x *= cohesionWeight;
        cohesionForce.y *= cohesionWeight;
        cohesionForce.z *= cohesionWeight;

        float sepMag = sqrtf(separationForce.x * separationForce.x +
            separationForce.y * separationForce.y +
            separationForce.z * separationForce.z);
        float aliMag = sqrtf(alignmentForce.x * alignmentForce.x +
            alignmentForce.y * alignmentForce.y +
            alignmentForce.z * alignmentForce.z);
        float cohMag = sqrtf(cohesionForce.x * cohesionForce.x +
            cohesionForce.y * cohesionForce.y +
            cohesionForce.z * cohesionForce.z);

        if (separationWeight < EPSILON) sepMag = 0.0f;
        if (alignmentWeight < EPSILON) aliMag = 0.0f;
        if (cohesionWeight < EPSILON) cohMag = 0.0f;

        if (sepMag > aliMag && sepMag > cohMag) {
            bird.dominantForce = 0;  // Separation (red)
        }
        else if (aliMag > sepMag && aliMag > cohMag) {
            bird.dominantForce = 1;  // Alignment (blue)
        }
        else if (cohMag > 0.0f) {
            bird.dominantForce = 2;  // Cohesion (green)
        }
        else {
            bird.dominantForce = 0;
        }

        bird.acceleration.x = separationForce.x + alignmentForce.x + cohesionForce.x;
        bird.acceleration.y = separationForce.y + alignmentForce.y + cohesionForce.y;
        bird.acceleration.z = separationForce.z + alignmentForce.z + cohesionForce.z;
    }
}

__global__ void updatePositions(float dt) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < numBirds) {
        Bird& bird = birds[tid];

        bird.velocity.x += bird.acceleration.x * dt;
        bird.velocity.y += bird.acceleration.y * dt;
        bird.velocity.z += bird.acceleration.z * dt;

        float speed = sqrtf(bird.velocity.x * bird.velocity.x +
            bird.velocity.y * bird.velocity.y +
            bird.velocity.z * bird.velocity.z);
        if (speed > MAX_SPEED) {
            bird.velocity.x = (bird.velocity.x / speed) * MAX_SPEED;
            bird.velocity.y = (bird.velocity.y / speed) * MAX_SPEED;
            bird.velocity.z = (bird.velocity.z / speed) * MAX_SPEED;
        }

        bird.position.x += bird.velocity.x * dt;
        bird.position.y += bird.velocity.y * dt;
        bird.position.z += bird.velocity.z * dt;

        bird.acceleration = make_float3(0.0f, 0.0f, 0.0f);

        if (bird.position.x < minBounds.x) bird.position.x = maxBounds.x;
        if (bird.position.y < minBounds.y) bird.position.y = maxBounds.y;
        if (bird.position.z < minBounds.z) bird.position.z = maxBounds.z;
        if (bird.position.x > maxBounds.x) bird.position.x = minBounds.x;
        if (bird.position.y > maxBounds.y) bird.position.y = minBounds.y;
        if (bird.position.z > maxBounds.z) bird.position.z = minBounds.z;
    }
}

struct Ray {
    float3 origin;
    float3 direction;
};

__global__ void renderBirdsKernel(uchar4* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float u = (float)x / width * 2.0f - 1.0f;
    float v = (float)(height - y) / height * 2.0f - 1.0f;

    u *= (float)width / height;

    float3 cameraPos = make_float3(0.0f, 0.0f, 120.0f);
    float3 lookAt = make_float3(0.0f, 0.0f, 0.0f);
    float3 up = make_float3(0.0f, 1.0f, 0.0f);

    float3 forward = make_float3(
        lookAt.x - cameraPos.x,
        lookAt.y - cameraPos.y,
        lookAt.z - cameraPos.z
    );
    float len = sqrtf(forward.x * forward.x + forward.y * forward.y + forward.z * forward.z);
    forward.x /= len;
    forward.y /= len;
    forward.z /= len;

    float3 right = make_float3(
        up.y * forward.z - up.z * forward.y,
        up.z * forward.x - up.x * forward.z,
        up.x * forward.y - up.y * forward.x
    );
    len = sqrtf(right.x * right.x + right.y * right.y + right.z * right.z);
    right.x /= len;
    right.y /= len;
    right.z /= len;

    float3 camUp = make_float3(
        forward.y * right.z - forward.z * right.y,
        forward.z * right.x - forward.x * right.z,
        forward.x * right.y - forward.y * right.x
    );

    float3 rayDirection = make_float3(
        forward.x + u * right.x + v * camUp.x,
        forward.y + u * right.y + v * camUp.y,
        forward.z + u * right.z + v * camUp.z
    );
    len = sqrtf(rayDirection.x * rayDirection.x + rayDirection.y * rayDirection.y + rayDirection.z * rayDirection.z);
    rayDirection.x /= len;
    rayDirection.y /= len;
    rayDirection.z /= len;

    Ray ray;
    ray.origin = cameraPos;
    ray.direction = rayDirection;

    uchar4 bgColor = make_uchar4(25, 25, 40, 255);
    output[y * width + x] = bgColor;

    const float birdRadius = 1.0f;
    float closestHit = 1e30f;
    int closestBird = -1;

    int maxBirdsToCheck = min(numBirds, MAX_BIRDS);

    for (int i = 0; i < maxBirdsToCheck; i++) {
        float3 oc = make_float3(
            ray.origin.x - birds[i].position.x,
            ray.origin.y - birds[i].position.y,
            ray.origin.z - birds[i].position.z
        );

        float a = ray.direction.x * ray.direction.x +
            ray.direction.y * ray.direction.y +
            ray.direction.z * ray.direction.z;
        float b = 2.0f * (oc.x * ray.direction.x +
            oc.y * ray.direction.y +
            oc.z * ray.direction.z);
        float c = oc.x * oc.x + oc.y * oc.y + oc.z * oc.z - birdRadius * birdRadius;

        float discriminant = b * b - 4 * a * c;

        if (discriminant > 0) {
            float temp = (-b - sqrtf(discriminant)) / (2.0f * a);

            if (temp > 0.001f && temp < closestHit) {
                closestHit = temp;
                closestBird = i;
            }
        }
    }

    if (closestBird >= 0 && closestBird < maxBirdsToCheck) {
        uchar4 color;

        switch (birds[closestBird].dominantForce) {
        case 0: // Separation = Red
            color = make_uchar4(230, 50, 50, 255);
            break;
        case 1: // Alignment = Blue
            color = make_uchar4(50, 50, 230, 255);
            break;
        case 2: // Cohesion = Green
            color = make_uchar4(50, 230, 50, 255);
            break;
        default:
            color = make_uchar4(200, 200, 200, 255);
        }

        float3 hitPoint = make_float3(
            ray.origin.x + closestHit * ray.direction.x,
            ray.origin.y + closestHit * ray.direction.y,
            ray.origin.z + closestHit * ray.direction.z
        );

        float3 normal = make_float3(
            hitPoint.x - birds[closestBird].position.x,
            hitPoint.y - birds[closestBird].position.y,
            hitPoint.z - birds[closestBird].position.z
        );

        len = sqrtf(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
        if (len > EPSILON) {
            normal.x /= len;
            normal.y /= len;
            normal.z /= len;
        }

        float3 lightDir = make_float3(0.5f, 0.5f, 1.0f);
        len = sqrtf(lightDir.x * lightDir.x + lightDir.y * lightDir.y + lightDir.z * lightDir.z);
        lightDir.x /= len;
        lightDir.y /= len;
        lightDir.z /= len;

        float diffuse = normal.x * lightDir.x + normal.y * lightDir.y + normal.z * lightDir.z;
        diffuse = max(0.2f, diffuse);

        color.x = min(255, (int)(color.x * diffuse));
        color.y = min(255, (int)(color.y * diffuse));
        color.z = min(255, (int)(color.z * diffuse));

        output[y * width + x] = color;
    }
}

__global__ void initMetrics() {
    metrics.forceCalculationTime = 0.0f;
    metrics.positionUpdateTime = 0.0f;
    metrics.stepsCompleted = 0;
    metrics.totalTime = 0.0f;
}

__global__ void resetMetrics() {
    metrics.forceCalculationTime = 0.0f;
    metrics.positionUpdateTime = 0.0f;
    metrics.stepsCompleted = 0;
    metrics.totalTime = 0.0f;
}

void freeCudaResources() {
    if (d_states) {
        cudaFree(d_states);
        d_states = nullptr;
    }

    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    cudaEventDestroy(forceStartEvent);
    cudaEventDestroy(forceStopEvent);
    cudaEventDestroy(posStartEvent);
    cudaEventDestroy(posStopEvent);
}

extern "C" void initSimulation(int birdCount, float* minBoundsArray, float* maxBoundsArray) {
    if (birdCount > MAX_BIRDS) {
        printf("Warning: Requested bird count %d exceeds maximum of %d. Using %d birds.\n",
            birdCount, MAX_BIRDS, MAX_BIRDS);
        birdCount = MAX_BIRDS;
    }

    cudaMemcpyToSymbol(numBirds, &birdCount, sizeof(int));

    float3 mins = make_float3(minBoundsArray[0], minBoundsArray[1], minBoundsArray[2]);
    float3 maxs = make_float3(maxBoundsArray[0], maxBoundsArray[1], maxBoundsArray[2]);

    cudaMemcpyToSymbol(minBounds, &mins, sizeof(float3));
    cudaMemcpyToSymbol(maxBounds, &maxs, sizeof(float3));

    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    cudaEventCreate(&forceStartEvent);
    cudaEventCreate(&forceStopEvent);
    cudaEventCreate(&posStartEvent);
    cudaEventCreate(&posStopEvent);

    cudaMalloc(&d_states, birdCount * sizeof(curandState));

    dim3 blockSize(256);
    dim3 gridSize((birdCount + blockSize.x - 1) / blockSize.x);

    initMetrics << <1, 1 >> > ();
    cudaDeviceSynchronize();

    setupRNG << <gridSize, blockSize >> > (d_states);
    cudaDeviceSynchronize();

    initBirds << <gridSize, blockSize >> > (d_states);
    cudaDeviceSynchronize();

    printf("Simulation initialized with %d birds\n", birdCount);
}

extern "C" void updateSimulation(float dt, float separationWeight, float alignmentWeight, float cohesionWeight) {
    int birdCount;
    cudaMemcpyFromSymbol(&birdCount, numBirds, sizeof(int));

    dim3 blockSize(256);
    dim3 gridSize((birdCount + blockSize.x - 1) / blockSize.x);

    float elapsedTime;

    cudaEventRecord(startEvent);

    cudaEventRecord(forceStartEvent);
    calculateForces << <gridSize, blockSize, blockSize.x * sizeof(Bird) >> > (separationWeight, alignmentWeight, cohesionWeight);
    cudaEventRecord(forceStopEvent);
    cudaDeviceSynchronize();

    cudaEventRecord(posStartEvent);
    updatePositions << <gridSize, blockSize >> > (dt);
    cudaEventRecord(posStopEvent);
    cudaDeviceSynchronize();

    cudaEventRecord(stopEvent);
    cudaEventSynchronize(stopEvent);

    cudaEventElapsedTime(&elapsedTime, forceStartEvent, forceStopEvent);
    PerformanceMetrics h_metrics;
    cudaMemcpyFromSymbol(&h_metrics, metrics, sizeof(PerformanceMetrics));
    h_metrics.forceCalculationTime += elapsedTime / 1000.0f;

    cudaEventElapsedTime(&elapsedTime, posStartEvent, posStopEvent);
    h_metrics.positionUpdateTime += elapsedTime / 1000.0f;

    cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
    h_metrics.totalTime += elapsedTime / 1000.0f;

    h_metrics.stepsCompleted++;

    cudaMemcpyToSymbol(metrics, &h_metrics, sizeof(PerformanceMetrics));
}

extern "C" void runBenchmark(int birdCount, int steps, float dt, float separationWeight, float alignmentWeight, float cohesionWeight) {
    resetMetrics << <1, 1 >> > ();
    cudaDeviceSynchronize();

    printf("Running benchmark for %d birds with %d steps...\n", birdCount, steps);

    for (int i = 1; i <= steps; i++) {
        updateSimulation(dt, separationWeight, alignmentWeight, cohesionWeight);

        if (i % 100 == 0 || i == steps) {
            printf("Completed %d steps (%.1f%%)\n", i, (i * 100.0f) / steps);
        }
    }

    PerformanceMetrics h_metrics;
    cudaMemcpyFromSymbol(&h_metrics, metrics, sizeof(PerformanceMetrics));

    printf("\n=== Performance Report ===\n");
    printf("Total steps: %d\n", h_metrics.stepsCompleted);
    printf("Total time: %.3f seconds\n", h_metrics.totalTime);
    printf("Steps per second: %.1f\n", h_metrics.stepsCompleted / h_metrics.totalTime);
    printf("Time breakdown:\n");
    printf("  - Force calculations: %.3fs (%.1f%%)\n",
        h_metrics.forceCalculationTime,
        (h_metrics.forceCalculationTime * 100.0f) / h_metrics.totalTime);
    printf("  - Position updates: %.3fs (%.1f%%)\n",
        h_metrics.positionUpdateTime,
        (h_metrics.positionUpdateTime * 100.0f) / h_metrics.totalTime);

    float overhead = h_metrics.totalTime -
        (h_metrics.forceCalculationTime +
            h_metrics.positionUpdateTime);

    printf("  - Other/overhead: %.3fs (%.1f%%)\n",
        overhead,
        (overhead * 100.0f) / h_metrics.totalTime);
    printf("=========================\n");
}

extern "C" void runScalingTest(int* flockSizes, int numSizes, int steps, float dt, float separationWeight, float alignmentWeight, float cohesionWeight) {
    printf("Running scaling test with various flock sizes...\n");

    for (int s = 0; s < numSizes; s++) {
        int birdCount = flockSizes[s];
        printf("\n=== Testing with %d birds ===\n", birdCount);

        float minBoundsArray[3] = { -50.0f, -50.0f, -50.0f };
        float maxBoundsArray[3] = { 50.0f, 50.0f, 50.0f };

        freeCudaResources();

        initSimulation(birdCount, minBoundsArray, maxBoundsArray);

        runBenchmark(birdCount, steps, dt, separationWeight, alignmentWeight, cohesionWeight);
    }
}

extern "C" void renderBirds(int width, int height, uchar4* output) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y);

    renderBirdsKernel << <gridSize, blockSize >> > (output, width, height);
    cudaDeviceSynchronize();
}

extern "C" void getPerformanceMetrics(PerformanceMetrics* out_metrics) {
    cudaMemcpyFromSymbol(out_metrics, metrics, sizeof(PerformanceMetrics));
}

extern "C" void freeSimulation() {
    freeCudaResources();
}