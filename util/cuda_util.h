#include <cuda_runtime.h>

template <typename T>
__device__ T WarpReduceSum(T val)
{
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}
