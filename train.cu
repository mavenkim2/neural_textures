#include <cmath>
#include <cstdint>
#include <cuda.h>

#if defined(__CUDACC__)
#define NT_DEVICE __device__
#endif

static const int gWarpSize = 32;

namespace
{

void Something() {}

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

// 1. Sample latents (which also somehow involves BC6 decompress) and reference
//      a. Train for 5k iterations with "unconstrained parameters" which just means
//      normal float3s.
//      b. Then block commpress with BC6.
// 2. Propagate batch forward through MLP
// 3. Compute loss and loss gradients
// 4. Backpropagate
// 5. Adam/update weights
// 6. Update latents (since BC6 is differentiable)

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

struct RNG
{
    float UniformFloat()
    {
        return 0.f;
    }
};

struct BC6Parameters
{
    // float3 ;
    int partition;
};

NT_DEVICE static void UnquantizeEndpoints() {}

static const int gBC6HPartitionSets[] = {
    0xcccc,
};

NT_DEVICE void BackpropagateBC6(float dloss)
{
    float y;
    float x;                   // pixel index
    int partitionSetIndex = 0; // partition set index, 0..31
    int partitionSet = gBC6HPartitionSets[partitionSetIndex];
    int pixel;
    int partition = (partitionSet >> pixel) & 0x1;
    float ebar2, ebar1;
    float ebar3, ebar4;

    const int q = 3; // TODO: always 3?

    float hy = fmaxf(floorf((y - 1.f) / 1024.f), 0.f);
    float dwdy = (1u << (hy - 24));

    float alpha = (1 << q) * x;
    float dy_debar1 = partition * (1.f - alpha); //
    float dy_debar2 = partition * alpha;

    float dy_debar3 = (1 - partition) * (1.f - alpha); //
    float dy_debar4 = (1 - partition) * alpha;

    float dy_dx =
        partition * (1 << q) * (ebar2 - ebar1) + (1 - partition) * (1 << q) * (ebar4 - ebar3);

    const float a = 31.f / 64;
    float deibar_dei = a * (1 << (16 - b));

    float dy_de1 = dy_debar1 * deibar_dei;
}

NT_DEVICE void BackpropagateTrilinear()
{
    // 31/64 unsigned, 31/32 signed
    int b = 1;

    // Backpropagate through trilinear filtering
    float s = 0.f;     // ?
    float dloss = 0.f; // ?
    float2 uv = ? ;

    float du0;
    float dv0;

    float du1;
    float dv1;

    float sFrac = s - floorf(s);

    // ts0 means closest lower mip. ts1 means closest higher mip
    float dloss_ts0_00 = (1.f - sFrac) * (1 - du0) * (1 - dv0) * dloss;
    float dloss_ts0_10 = (1.f - sFrac) * du0 * (1 - dv0) * dloss;
    float dloss_ts0_01 = (1.f - sFrac) * (1 - du0) * dv0 * dloss;
    float dloss_ts0_11 = (1.f - sFrac) * du0 * dv0 * dloss;

    float dloss_ts1_00 = sFrac * (1 - du1) * (1 - dv1) * dloss;
    float dloss_ts1_10 = sFrac * du1 * (1 - dv1) * dloss;
    float dloss_ts1_01 = sFrac * (1 - du1) * dv1 * dloss;
    float dloss_ts1_11 = sFrac * du1 * dv1 * dloss;

    // Backpropagate to BC6 format

    float dloss_over_dei = dloss_over_dei_bar * dei_bar_over_dei;
}

NT_DEVICE DecompressBC6() {}

NT_DEVICE TrainLoop()
{
    RNG rng;
    for (int iter = 0; iter < NUM_ITERS; iter++)
    {
        float u = rng.UniformFloat();
    }
}

struct Layer
{
    // 32-byte aligned
    half *weights;
};

} // namespace
