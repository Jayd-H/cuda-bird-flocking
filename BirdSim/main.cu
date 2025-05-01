#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <helper_math.h>
#include <helper_cuda.h>
#include <curand_kernel.h>

#define MAX_BIRDS 10000
#define EPSILON 1e-6f
#define PERCEPTION_RADIUS 15.0f
#define SEPARATION_RADIUS_FACTOR 0.3f
#define MAX_SPEED 3.0f
#define MAX_FORCE 0.1f

typedef unsigned char uchar;

// Bird struct that will be used on the GPU
struct Bird {
    float3 position;
    float3 velocity;
    float3 acceleration;
    int dominantForce; // 0 = separation, 1 = alignment, 2 = cohesion
};

// Global variables for the simulation
__device__ __managed__ Bird birds[MAX_BIRDS];
__device__ __managed__ int numBirds;
__device__ __managed__ float3 minBounds;
__device__ __managed__ float3 maxBounds;

// Initialize random number generator
__global__ void setupRNG(curandState* states) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < numBirds) {
        curand_init(clock64(), tid, 0, &states[tid]);
    }
}

// Initialize birds with random positions and velocities
__global__ void initBirds(curandState* states) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < numBirds) {
        curandState localState = states[tid];

        // Random position within bounds
        birds[tid].position.x = minBounds.x + curand_uniform(&localState) * (maxBounds.x - minBounds.x);
        birds[tid].position.y = minBounds.y + curand_uniform(&localState) * (maxBounds.y - minBounds.y);
        birds[tid].position.z = minBounds.z + curand_uniform(&localState) * (maxBounds.z - minBounds.z);

        // Random velocity (-1 to 1)
        float3 vel;
        vel.x = curand_uniform(&localState) * 2.0f - 1.0f;
        vel.y = curand_uniform(&localState) * 2.0f - 1.0f;
        vel.z = curand_uniform(&localState) * 2.0f - 1.0f;

        // Normalize velocity and scale
        float len = sqrtf(vel.x * vel.x + vel.y * vel.y + vel.z * vel.z);
        if (len > EPSILON) {
            vel.x /= len;
            vel.y /= len;
            vel.z /= len;
        }

        // Scale velocity and set initial acceleration to zero
        float speed = 0.5f + curand_uniform(&localState) * 1.5f; // Between 0.5 and 2.0
        birds[tid].velocity = make_float3(vel.x * speed, vel.y * speed, vel.z * speed);
        birds[tid].acceleration = make_float3(0.0f, 0.0f, 0.0f);
        birds[tid].dominantForce = 0;

        // Save state back for next call
        states[tid] = localState;
    }
}

// Calculate the wrapped distance between two positions
__device__ float3 calculateWrappedDistance(float3 pos1, float3 pos2) {
    float3 diff = make_float3(pos1.x - pos2.x, pos1.y - pos2.y, pos1.z - pos2.z);

    float size_x = maxBounds.x - minBounds.x;
    float size_y = maxBounds.y - minBounds.y;
    float size_z = maxBounds.z - minBounds.z;

    // Handle wraparound (toroidal space)
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

// Calculate flocking forces for each bird
__global__ void calculateForces(float separationWeight, float alignmentWeight, float cohesionWeight) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < numBirds) {
        Bird& bird = birds[tid];

        float3 separation = make_float3(0.0f, 0.0f, 0.0f);
        float3 alignment = make_float3(0.0f, 0.0f, 0.0f);
        float3 cohesion = make_float3(0.0f, 0.0f, 0.0f);
        int separationCount = 0;
        int alignmentCount = 0;
        int cohesionCount = 0;

        // Check against all other birds
        for (int i = 0; i < numBirds; i++) {
            if (i == tid) continue;

            Bird& other = birds[i];
            float3 diff = calculateWrappedDistance(bird.position, other.position);
            float dist_sq = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;

            // Separation (avoid crowding neighbors)
            if (dist_sq > EPSILON && dist_sq < (PERCEPTION_RADIUS * SEPARATION_RADIUS_FACTOR) * (PERCEPTION_RADIUS * SEPARATION_RADIUS_FACTOR)) {
                // Calculate separation vector - points away from neighbor
                float len = sqrtf(dist_sq);
                float3 repulse = make_float3(diff.x / dist_sq, diff.y / dist_sq, diff.z / dist_sq);
                separation.x += repulse.x;
                separation.y += repulse.y;
                separation.z += repulse.z;
                separationCount++;
            }

            // For both alignment and cohesion we use the full perception radius
            if (dist_sq < PERCEPTION_RADIUS * PERCEPTION_RADIUS) {
                // Alignment (steer towards average heading of neighbors)
                alignment.x += other.velocity.x;
                alignment.y += other.velocity.y;
                alignment.z += other.velocity.z;
                alignmentCount++;

                // Cohesion (move toward average position of neighbors)
                // Calculate wrapped position of the other bird
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

        // Compute final forces with weights
        float3 separationForce = make_float3(0.0f, 0.0f, 0.0f);
        float3 alignmentForce = make_float3(0.0f, 0.0f, 0.0f);
        float3 cohesionForce = make_float3(0.0f, 0.0f, 0.0f);

        // Finalize separation force
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

        // Finalize alignment force
        if (alignmentCount > 0) {
            alignment.x /= alignmentCount;
            alignment.y /= alignmentCount;
            alignment.z /= alignmentCount;

            // Calculate steering force = desired - current
            float3 steer = make_float3(
                alignment.x - bird.velocity.x,
                alignment.y - bird.velocity.y,
                alignment.z - bird.velocity.z
            );

            // Limit the force
            float len = sqrtf(steer.x * steer.x + steer.y * steer.y + steer.z * steer.z);
            if (len > MAX_FORCE && len > EPSILON) {
                steer.x = (steer.x / len) * MAX_FORCE;
                steer.y = (steer.y / len) * MAX_FORCE;
                steer.z = (steer.z / len) * MAX_FORCE;
            }

            alignmentForce = steer;
        }

        // Finalize cohesion force
        if (cohesionCount > 0) {
            cohesion.x /= cohesionCount;
            cohesion.y /= cohesionCount;
            cohesion.z /= cohesionCount;

            // Create vector pointing from current position to target
            float3 desired = make_float3(
                cohesion.x - bird.position.x,
                cohesion.y - bird.position.y,
                cohesion.z - bird.position.z
            );

            // Scale to max speed
            float len = sqrtf(desired.x * desired.x + desired.y * desired.y + desired.z * desired.z);
            if (len > EPSILON) {
                desired.x = (desired.x / len) * MAX_SPEED;
                desired.y = (desired.y / len) * MAX_SPEED;
                desired.z = (desired.z / len) * MAX_SPEED;
            }

            // Steering = Desired - Current
            float3 steer = make_float3(
                desired.x - bird.velocity.x,
                desired.y - bird.velocity.y,
                desired.z - bird.velocity.z
            );

            // Limit the force
            len = sqrtf(steer.x * steer.x + steer.y * steer.y + steer.z * steer.z);
            if (len > MAX_FORCE && len > EPSILON) {
                steer.x = (steer.x / len) * MAX_FORCE;
                steer.y = (steer.y / len) * MAX_FORCE;
                steer.z = (steer.z / len) * MAX_FORCE;
            }

            cohesionForce = steer;
        }

        // Apply weights to forces
        separationForce.x *= separationWeight;
        separationForce.y *= separationWeight;
        separationForce.z *= separationWeight;

        alignmentForce.x *= alignmentWeight;
        alignmentForce.y *= alignmentWeight;
        alignmentForce.z *= alignmentWeight;

        cohesionForce.x *= cohesionWeight;
        cohesionForce.y *= cohesionWeight;
        cohesionForce.z *= cohesionWeight;

        // Determine dominant force for coloring
        float sepMag = sqrtf(separationForce.x * separationForce.x +
            separationForce.y * separationForce.y +
            separationForce.z * separationForce.z);
        float aliMag = sqrtf(alignmentForce.x * alignmentForce.x +
            alignmentForce.y * alignmentForce.y +
            alignmentForce.z * alignmentForce.z);
        float cohMag = sqrtf(cohesionForce.x * cohesionForce.x +
            cohesionForce.y * cohesionForce.y +
            cohesionForce.z * cohesionForce.z);

        if (sepMag > aliMag && sepMag > cohMag) {
            bird.dominantForce = 0;  // Separation dominant
        }
        else if (aliMag > sepMag && aliMag > cohMag) {
            bird.dominantForce = 1;  // Alignment dominant
        }
        else {
            bird.dominantForce = 2;  // Cohesion dominant
        }

        // Sum all forces
        bird.acceleration.x = separationForce.x + alignmentForce.x + cohesionForce.x;
        bird.acceleration.y = separationForce.y + alignmentForce.y + cohesionForce.y;
        bird.acceleration.z = separationForce.z + alignmentForce.z + cohesionForce.z;
    }
}

// Update bird positions based on velocity and acceleration
__global__ void updatePositions(float dt) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < numBirds) {
        Bird& bird = birds[tid];

        // Update velocity by acceleration
        bird.velocity.x += bird.acceleration.x * dt;
        bird.velocity.y += bird.acceleration.y * dt;
        bird.velocity.z += bird.acceleration.z * dt;

        // Limit speed to maximum
        float speed = sqrtf(bird.velocity.x * bird.velocity.x +
            bird.velocity.y * bird.velocity.y +
            bird.velocity.z * bird.velocity.z);
        if (speed > MAX_SPEED) {
            bird.velocity.x = (bird.velocity.x / speed) * MAX_SPEED;
            bird.velocity.y = (bird.velocity.y / speed) * MAX_SPEED;
            bird.velocity.z = (bird.velocity.z / speed) * MAX_SPEED;
        }

        // Update position by velocity
        bird.position.x += bird.velocity.x * dt;
        bird.position.y += bird.velocity.y * dt;
        bird.position.z += bird.velocity.z * dt;

        // Reset acceleration
        bird.acceleration = make_float3(0.0f, 0.0f, 0.0f);

        // Handle boundary conditions (wraparound)
        if (bird.position.x < minBounds.x) bird.position.x = maxBounds.x;
        if (bird.position.y < minBounds.y) bird.position.y = maxBounds.y;
        if (bird.position.z < minBounds.z) bird.position.z = maxBounds.z;
        if (bird.position.x > maxBounds.x) bird.position.x = minBounds.x;
        if (bird.position.y > maxBounds.y) bird.position.y = minBounds.y;
        if (bird.position.z > maxBounds.z) bird.position.z = minBounds.z;
    }
}

// Ray struct for rendering
struct Ray {
    float3 origin;
    float3 direction;
};

// Function to render birds using simple ray casting
__global__ void renderBirdsKernel(uchar4* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Calculate normalized device coordinates
    float u = (float)x / width * 2.0f - 1.0f;
    float v = (float)(height - y) / height * 2.0f - 1.0f;

    // Maintain aspect ratio
    u *= (float)width / height;

    // Camera setup
    float3 cameraPos = make_float3(0.0f, 0.0f, 120.0f);
    float3 lookAt = make_float3(0.0f, 0.0f, 0.0f);
    float3 up = make_float3(0.0f, 1.0f, 0.0f);

    // Create normalized camera direction vectors
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

    // Create ray direction
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

    // Background color
    uchar4 bgColor = make_uchar4(25, 25, 40, 255);
    output[y * width + x] = bgColor;

    // Render each bird as a simple colored sphere
    const float birdRadius = 1.0f;
    float closestHit = 1e30f;
    int closestBird = -1;

    for (int i = 0; i < numBirds; i++) {
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

    // If we hit a bird, color it based on its dominant force
    if (closestBird >= 0) {
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

        // Simple lighting
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
        normal.x /= len;
        normal.y /= len;
        normal.z /= len;

        float3 lightDir = make_float3(0.5f, 0.5f, 1.0f);
        len = sqrtf(lightDir.x * lightDir.x + lightDir.y * lightDir.y + lightDir.z * lightDir.z);
        lightDir.x /= len;
        lightDir.y /= len;
        lightDir.z /= len;

        float diffuse = normal.x * lightDir.x + normal.y * lightDir.y + normal.z * lightDir.z;
        diffuse = max(0.2f, diffuse); // Ambient + diffuse

        color.x = min(255, (int)(color.x * diffuse));
        color.y = min(255, (int)(color.y * diffuse));
        color.z = min(255, (int)(color.z * diffuse));

        output[y * width + x] = color;
    }
}

// CUDA resources
curandState* d_states = nullptr;

// Initialize the simulation
extern "C" void initSimulation(int birdCount, float* minBoundsArray, float* maxBoundsArray) {
    // Set up simulation parameters
    cudaMemcpyToSymbol(numBirds, &birdCount, sizeof(int));

    float3 mins = make_float3(minBoundsArray[0], minBoundsArray[1], minBoundsArray[2]);
    float3 maxs = make_float3(maxBoundsArray[0], maxBoundsArray[1], maxBoundsArray[2]);

    cudaMemcpyToSymbol(minBounds, &mins, sizeof(float3));
    cudaMemcpyToSymbol(maxBounds, &maxs, sizeof(float3));

    // Allocate random state memory
    cudaMalloc(&d_states, birdCount * sizeof(curandState));

    // Calculate kernel launch parameters
    dim3 blockSize(256);
    dim3 gridSize((birdCount + blockSize.x - 1) / blockSize.x);

    // Initialize random states and birds
    setupRNG << <gridSize, blockSize >> > (d_states);
    cudaDeviceSynchronize();

    initBirds << <gridSize, blockSize >> > (d_states);
    cudaDeviceSynchronize();

    printf("Simulation initialized with %d birds\n", birdCount);
}

// Update the simulation for one time step
extern "C" void updateSimulation(float dt, float separationWeight, float alignmentWeight, float cohesionWeight) {
    dim3 blockSize(256);
    dim3 gridSize((numBirds + blockSize.x - 1) / blockSize.x);

    calculateForces << <gridSize, blockSize >> > (separationWeight, alignmentWeight, cohesionWeight);
    cudaDeviceSynchronize();

    updatePositions << <gridSize, blockSize >> > (dt);
    cudaDeviceSynchronize();
}

// Render the birds to the output buffer
extern "C" void renderBirds(int width, int height, uchar4* output) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y);

    renderBirdsKernel << <gridSize, blockSize >> > (output, width, height);
    cudaDeviceSynchronize();
}

// Free simulation resources
extern "C" void freeSimulation() {
    if (d_states) {
        cudaFree(d_states);
        d_states = nullptr;
    }
}