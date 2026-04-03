#pragma once

#include <cuda_runtime.h>

#if defined(__CUDACC__)
#define NT_HOST_DEVICE __host__ __device__
#else
#define NT_HOST_DEVICE
#endif

NT_HOST_DEVICE inline int3 make_int3(const float3 &v)
{
    return make_int3(int(v.x), int(v.y), int(v.z));
}

NT_HOST_DEVICE inline int3 operator-(const int3 &value, int scalar)
{
    return make_int3(value.x - scalar, value.y - scalar, value.z - scalar);
}
