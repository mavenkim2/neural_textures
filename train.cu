#include <cstdint>
#include <cuda.h>

#if defined(__CUDACC__)
#define NT_DEVICE __device__
#endif

static const int gWarpSize = 32;

namespace
{

template <typename T, int sizeA, int sizeB>
NT_DEVICE void
OuterProduct(T *__restrict__ result, const T *__restrict__ a, const T *__restrict__ b)
{
    static_assert(sizeA <= gWarpSize);
    static_assert(sizeB <= gWarpSize);

    uint32_t threadID = threadIdx.x;
    T val = threadID < sizeB ? b[threadID] : T(0);

    for (int i = 0; i < sizeA; i++)
    {
        T aVal = a[i];
        if (threadID < sizeB)
        {
            result[sizeB * i + threadID] = aVal * val;
        }
    }
}

template <typename T, int m, int n>
NT_DEVICE void MatrixMultiply(T *__restrict__ result, T *__restrict__ a, T *__restrict__ b)
{
    static_assert(m <= gWarpSize);
    static_assert(n <= gWarpSize);
}

template <typename T, int inputSize, int outputSize>
NT_DEVICE void Backward(const T *__restrict__ outGrad,
                        const T *__restrict__ inputs,
                        const T *__restrict__ weights,
                        T *__restrict__ weightGradient,
                        T *__restrict__ inputGradient)
{
    MatrixMultiply<T, inputSize, outputSize>(inputGradient, weights, outGrad);
    OuterProduct<T, inputSize, outputSize>(weightGradient, inputs, outGrad);
}

struct Layer
{
    // 32-byte aligned
    half *weights;
};

} // namespace
