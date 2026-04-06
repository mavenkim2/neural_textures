#include <cuda_runtime.h>

inline __device__ float3 atomicAdd(float3 *a, const float3 &b)
{
    float3 result;
    result.x = atomicAdd(&a->x, b.x);
    result.y = atomicAdd(&a->y, b.y);
    result.z = atomicAdd(&a->z, b.z);
    return result;
}
