#include <cmath>
#include <cstdint>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "util/float2.h"
#include "util/float3.h"

#include <tiny-cuda-nn/mma.h>

#if defined(__CUDACC__)
#define NT_DEVICE __device__
#endif

#define WARP_SIZE 32
static const int gNumFeatures = 4;
static const int gNumBC6PixelsPerBlock = 16;
static const int gNumIters = 5000;

#define half __half

namespace neural_textures
{

// clang-format off
NT_DEVICE static const int gBC6HPartitionSets[] = {
    0xCCCC, 0x8888, 0xEEEE, 0xECC8,
    0xC880, 0xFEEC, 0xFEC8, 0xEC80,
    0xC800, 0xFFEC, 0xFE80, 0xE800,
    0xFFE8, 0xFF00, 0xFFF0, 0xF000,
    0xF710, 0x008E, 0x7100, 0x08CE,
    0x008C, 0x7310, 0x3100, 0x8CCE,
    0x088C, 0x3110, 0x6666, 0x366C,
    0x17E8, 0x0FF0, 0x718E, 0x399C,
};

NT_DEVICE static const int gBC6HPartitionNumBits[] = 
{
    10, 7, 11, 11, 11, 9, 8, 8, 8, 6,
};
// clang-format on

struct RNG
{
    NT_DEVICE float UniformFloat()
    {
        return 0.f;
    }
};

// template <typename T, int sizeA, int sizeB>
// NT_DEVICE void
// OuterProduct(T *__restrict__ result, const T *__restrict__ a, const T *__restrict__ b)
// {
//     static_assert(sizeA <= gWarpSize);
//     static_assert(sizeB <= gWarpSize);
//
//     uint32_t threadID = threadIdx.x;
//     T val = threadID < sizeB ? b[threadID] : T(0);
//
//     for (int i = 0; i < sizeA; i++)
//     {
//         T aVal = a[i];
//         if (threadID < sizeB)
//         {
//             result[sizeB * i + threadID] = aVal * val;
//         }
//     }
// }

// template <typename T, int m, int n>
// NT_DEVICE void MatrixMultiply(T *result, T *a, T *b)
// {
//     static_assert(m <= gWarpSize);
//     static_assert(n <= gWarpSize);
// }
//
// template <typename T, int inputSize, int outputSize>
// NT_DEVICE void Backward(const T *__restrict__ outGrad,
//                         const T *__restrict__ inputs,
//                         const T *__restrict__ weights,
//                         T *__restrict__ weightGradient,
//                         T *__restrict__ inputGradient)
// {
//     MatrixMultiply<T, inputSize, outputSize>(inputGradient, weights, outGrad);
//     OuterProduct<T, inputSize, outputSize>(weightGradient, inputs, outGrad);
// }

// NT_DEVICE void BackpropagateBC6(float dloss)
// {
//     float y;
//     float x;                   // pixel index
//     int partitionSetIndex = 0; // partition set index, 0..31
//     int partitionSet = gBC6HPartitionSets[partitionSetIndex];
//     int pixel;
//     int partition = (partitionSet >> pixel) & 0x1;
//     float ebar2, ebar1;
//     float ebar3, ebar4;
//
//     const int q = 3; // TODO: always 3?
//
//     float hy = fmaxf(floorf((y - 1.f) / 1024.f), 0.f);
//     float dwdy = (1u << (hy - 24));
//
//     float alpha = (1 << q) * x;
//     float dy_debar1 = partition * (1.f - alpha); //
//     float dy_debar2 = partition * alpha;
//
//     float dy_debar3 = (1 - partition) * (1.f - alpha); //
//     float dy_debar4 = (1 - partition) * alpha;
//
//     float dy_dx =
//         partition * (1 << q) * (ebar2 - ebar1) + (1 - partition) * (1 << q) * (ebar4 - ebar3);
//
//     const float a = 31.f / 64;
//     float deibar_dei = a * (1 << (16 - b));
//
//     float dy_de1 = dy_debar1 * deibar_dei;
// }

// NT_DEVICE void BackpropagateTrilinear()
// {
//     // 31/64 unsigned, 31/32 signed
//     int b = 1;
//
//     // Backpropagate through trilinear filtering
//     float s = 0.f;     // ?
//     float dloss = 0.f; // ?
//     float2 uv = ? ;
//
//     float du0;
//     float dv0;
//
//     float du1;
//     float dv1;
//
//     float sFrac = s - floorf(s);
//
//     // ts0 means closest lower mip. ts1 means closest higher mip
//     float dloss_ts0_00 = (1.f - sFrac) * (1 - du0) * (1 - dv0) * dloss;
//     float dloss_ts0_10 = (1.f - sFrac) * du0 * (1 - dv0) * dloss;
//     float dloss_ts0_01 = (1.f - sFrac) * (1 - du0) * dv0 * dloss;
//     float dloss_ts0_11 = (1.f - sFrac) * du0 * dv0 * dloss;
//
//     float dloss_ts1_00 = sFrac * (1 - du1) * (1 - dv1) * dloss;
//     float dloss_ts1_10 = sFrac * du1 * (1 - dv1) * dloss;
//     float dloss_ts1_01 = sFrac * (1 - du1) * dv1 * dloss;
//     float dloss_ts1_11 = sFrac * du1 * dv1 * dloss;
//
//     // Backpropagate to BC6 format
//
//     float dloss_over_dei = dloss_over_dei_bar * dei_bar_over_dei;
// }

struct BC6Parameters
{
    float3 endpoints[4];
    float pixelIndices[gNumBC6PixelsPerBlock]; // [0, 1]
    int partition;
    int mode;
};

static const int gNumNetworkLayers = 2;

#define NT_INPUT_SIZE 12
#define NT_HIDDEN_LAYER_SIZE 16

struct KernelParams
{
    // params[mip][v * numU + u]
    BC6Parameters **t0;
    BC6Parameters **t1;
    BC6Parameters **t2;
    BC6Parameters **t3;

    half *networkWeights[gNumNetworkLayers];

    int imageWidth;
    int imageHeight;
    int numMips;
    int numBlocksU;
    int numBlocksV;
};

NT_DEVICE inline float Clamp(float x, float low, float high)
{
    return x < low ? low : (x > high ? high : x);
}

NT_DEVICE inline int WrapTexelCoord(int coord, int size)
{
    int wrapped = coord % size;
    return wrapped < 0 ? wrapped + size : wrapped;
}

NT_DEVICE inline float BilinearWeight(const int2 &corner, const float2 &texelFrac)
{
    float wu = corner.x ? texelFrac.x : (1.f - texelFrac.x);
    float wv = corner.y ? texelFrac.y : (1.f - texelFrac.y);
    return wu * wv;
}

// Endpoints are train-time float parameters, then mapped through the BC6-compatible
// endpoint quantization/dequantization transform before interpolation.
NT_DEVICE inline float3 UnquantizeEndpoints(const float3 &endpoint, int b)
{
    const float unsignedBC6Constant = 31.f / 64.f;
    const float scale = unsignedBC6Constant * 65536.f;

    return Ldexp(scale * endpoint + 32768.f, -b);
}

NT_DEVICE inline float3
DecompressBC6(const float3 &endpoint0, const float3 &endpoint1, int b, float x)
{
    const int q = 3;
    float3 unquantizedEndpoint0 = UnquantizeEndpoints(endpoint0, b);
    float3 unquantizedEndpoint1 = UnquantizeEndpoints(endpoint1, b);

    float3 y = unquantizedEndpoint0 + Ldexp(unquantizedEndpoint1 - unquantizedEndpoint0, q) * x;
    float3 hyFloat = Max(Floor((y - 1.f) / 1024.f) - 1.f, 0.f);
    int3 hy = make_int3(hyFloat);

    float3 w = Ldexp(y / 1024.f - make_float3(hy), hy - 14);
    return w;
}

NT_DEVICE inline float3 SampleBC6FeatureTexel(const BC6Parameters *texture,
                                              int numBlocksU,
                                              int imageWidth,
                                              int imageHeight,
                                              int texelX,
                                              int texelY)
{
    int paramIndex = (texelY >> 2) * numBlocksU + (texelX >> 2);
    int pixelIndex = ((texelY & 0x3) << 2) + (texelX & 0x3);

    const BC6Parameters &bc6Params = texture[paramIndex];
    int mask = gBC6HPartitionSets[bc6Params.partition];
    int partition = (mask >> pixelIndex) & 0x1;

    int endpointBase = 2 * partition;
    float3 endpoint0 = bc6Params.endpoints[endpointBase + 0];
    float3 endpoint1 = bc6Params.endpoints[endpointBase + 1];
    float pixelIndexOnSegment = bc6Params.pixelIndices[pixelIndex];

    const int endpointBits = gBC6HPartitionNumBits[bc6Params.mode];
    return DecompressBC6(endpoint0, endpoint1, endpointBits, pixelIndexOnSegment);
}

NT_DEVICE inline void
SampleFourFeaturesTrilinear(const KernelParams &params, float3 uvs, half *outFeatures)
{
    const BC6Parameters *const *textures[gNumFeatures] = {
        params.t0,
        params.t1,
        params.t2,
        params.t3,
    };

    int mipBase = (int)floorf(uvs.z);
    float mipFrac = uvs.z - floorf(uvs.z);

    for (int featureIndex = 0; featureIndex < gNumFeatures * 3; ++featureIndex)
    {
        outFeatures[featureIndex] = 0;
    }

    // TODO: consider reordering loads/global memory accesses
    for (int mipIndex = 0; mipIndex <= 1; mipIndex++)
    {
        int mip = mipBase + mipIndex;
        int width = max(params.imageWidth >> mip, 1);
        int height = max(params.imageHeight >> mip, 1);
        int numBlocksU = max(1, width >> 2);

        float2 imageSize = make_float2(float(width), float(height));
        float2 texelUV = make_float2(uvs.x, uvs.y) * imageSize - 0.5f;
        float2 texelBaseUV = Floor(texelUV);
        int2 texelBase = make_int2(int(texelBaseUV.x), int(texelBaseUV.y));
        float2 texelFrac = texelUV - texelBaseUV;

        // TODO: unroll?
        for (int corner = 0; corner < 4; corner++)
        {
            int cornerX = corner & 1;
            int cornerY = corner >> 1;

            int texelX = texelBase.x + cornerX;
            int texelY = texelBase.y + cornerY;
            texelX = WrapTexelCoord(texelX, width);
            texelY = WrapTexelCoord(texelY, height);
            float weight = (mipIndex ? mipFrac : 1.f - mipFrac) *
                           BilinearWeight(make_int2(cornerX, cornerY), texelFrac);

            for (int featureIndex = 0; featureIndex < gNumFeatures; ++featureIndex)
            {
                const BC6Parameters *texture = textures[featureIndex][mip];
                float3 feature = weight * SampleBC6FeatureTexel(
                                              texture, numBlocksU, width, height, texelX, texelY);

                outFeatures[3 * featureIndex] += feature.x;
                outFeatures[3 * featureIndex + 1] += feature.y;
                outFeatures[3 * featureIndex + 2] += feature.z;
            }
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
NT_DEVICE void TrainLoop(const KernelParams params)
{
    const uint32_t threadID = threadIdx.y * blockDim.x + threadIdx.x;
    const uint32_t lane = threadID & (WARP_SIZE - 1u);
    RNG rng;

    for (int iter = 0; iter < gNumIters; iter++)
    {
        float u = rng.UniformFloat();
        float v = rng.UniformFloat();
        float s = rng.UniformFloat();
        (void)s;

        half sampledFeatures[gNumFeatures * 3];
        SampleFourFeaturesTrilinear(params, make_float3(u, v, s), sampledFeatures);

        // TODO: feed the four sampled feature vectors into the MLP, compare against the
        // filtered reference material sample, and backpropagate into network + BC6 params.

        // TODO: sample the reference textures

        // Forward pass
        // Hidden layer
        tcnn::hvec<NT_INPUT_SIZE> inputsVector0(sampledFeatures);
        tcnn::mma_mat<WARP_SIZE, 16, tcnn::RM> inputsMatrix0(inputsVector0);

        tcnn::mma_mat<16, 16, tcnn::CM> weightsHiddenLayer0 =
            tcnn::mma_mat<16, 16, tcnn::CM>::from_linear_memory(params.networkWeights[0]);

        auto outputHiddenLayer0 = inputsMatrix0 * weightsHiddenLayer0;
        tcnn::hvec<NT_HIDDEN_LAYER_SIZE> y0 = outputHiddenLayer0.vec<NT_HIDDEN_LAYER_SIZE>();
        outputHiddenLayer0.activate<tcnn::Activation::ReLU>();

        // Output layer
        tcnn::mma_mat<16, 16, tcnn::CM> weightsHiddenLayer1 =
            tcnn::mma_mat<16, 16, tcnn::CM>::from_linear_memory(params.networkWeights[1]);

        auto finalOutput = outputHiddenLayer0 * weightsHiddenLayer0;
        finalOutput.activate<tcnn::Activation::ReLU>();
    }
}

struct Layer
{
    // 32-byte aligned
    half *weights;
};

} // namespace neural_textures
