#ifndef RAYH
#define RAYH
#include "vec3.h"

class ray {
public:
    __device__  ray() {}
    __device__  ray(const vec3& a, const vec3& b) { O = a; Dir = b; }
    __device__  vec3 origin() const { return O; }
    __device__  vec3 direction() const { return Dir; }
    __device__  vec3 point_at_parameter(float t) const { return O + t * Dir; }

    vec3 O;
    vec3 Dir;
};

#endif
