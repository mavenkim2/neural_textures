#include <cmath>
#include <cstdint>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "mlp.h"

#include "util/float2.h"
#include "util/float3.h"

static const int gNumFeatures = 4;
static const int gNumBC6PixelsPerBlock = 16;
static const int gNumIters = 5000;

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

struct BC6Parameters
{
    float3 endpoints[4];
    float pixelIndices[gNumBC6PixelsPerBlock]; // [0, 1]
    int partition;
    int mode;
};

struct Feature
{
    BC6Parameters **grid;
    int width; // in texels
    int height;
};

static const int gNumNetworkLayers = 2;

struct KernelParams
{
    // params[mip][v * numU + u]
    Feature features[gNumFeatures];

    half *networkWeights[gNumNetworkLayers];

    __half2 *networkMovingAverages[gNumNetworkLayers];

    int imageWidth;
    int imageHeight;
    int numMips;
    int numBlocksU;
    int numBlocksV;

    int numSamples;
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

NT_DEVICE inline float3 DecompressBC6IntermediateValues(const float3 &endpoint0,
                                                        const float3 &endpoint1,
                                                        int b,
                                                        float x,
                                                        float3 &unquantizedEndpoint0,
                                                        float3 &unquantizedEndpoint1,
                                                        float3 &y)
{
    const int q = 3;
    unquantizedEndpoint0 = UnquantizeEndpoints(endpoint0, b);
    unquantizedEndpoint1 = UnquantizeEndpoints(endpoint1, b);

    y = unquantizedEndpoint0 + Ldexp(unquantizedEndpoint1 - unquantizedEndpoint0, q) * x;
    float3 hyFloat = Max(Floor((y - 1.f) / 1024.f) - 1.f, 0.f);
    int3 hy = make_int3(hyFloat);

    float3 w = Ldexp(y / 1024.f - make_float3(hy), hy - 14);
    return w;
}

NT_DEVICE inline int GetBlockParamIndex(int texelX, int texelY, int numBlocksU)
{
    return (texelY >> 2) * numBlocksU + (texelX >> 2);
}

NT_DEVICE inline int GetBlockPixelIndex(int texelX, int texelY)
{
    return ((texelY & 0x3) << 2) + (texelX & 0x3);
}

NT_DEVICE inline float3 SampleBC6FeatureTexel(const BC6Parameters *texture,
                                              int numBlocksU,
                                              int imageWidth,
                                              int imageHeight,
                                              int texelX,
                                              int texelY)
{
    int paramIndex = GetBlockParamIndex(texelX, texelY, numBlocksU);
    int pixelIndex = GetBlockPixelIndex(texelX, texelY);

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
    int mipBase = (int)floorf(uvs.z);
    float mipFrac = uvs.z - floorf(uvs.z);

    for (int featureIndex = 0; featureIndex < gNumFeatures * 3; ++featureIndex)
    {
        outFeatures[featureIndex] = 0;
    }

    // TODO: consider reordering loads/global memory accesses

    for (int featureIndex = 0; featureIndex < gNumFeatures; ++featureIndex)
    {
        const Feature &feature = params.features[featureIndex];
        for (int mipIndex = 0; mipIndex <= 1; mipIndex++)
        {
            int mip = mipBase + mipIndex;
            const BC6Parameters *texture = feature.grid[mip];

            int width = max(feature.width >> mip, 1);
            int height = max(feature.height >> mip, 1);
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

                float3 feature = weight * SampleBC6FeatureTexel(
                                              texture, numBlocksU, width, height, texelX, texelY);

                outFeatures[3 * featureIndex] += feature.x;
                outFeatures[3 * featureIndex + 1] += feature.y;
                outFeatures[3 * featureIndex + 2] += feature.z;
            }
        }
    }
}

NT_DEVICE inline void
BackwardFeaturePass(const KernelParams &params, float3 uvs, half *inputGradient)
{
    int mipBase = (int)floorf(uvs.z);
    float mipFrac = uvs.z - floorf(uvs.z);

    // TODO: consider reordering loads/global memory accesses
    for (int featureIndex = 0; featureIndex < gNumFeatures; ++featureIndex)
    {
        const Feature &feature = params.features[featureIndex];
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

                const int pixelIndex = GetBlockPixelIndex(texelX, texelY);
                const int paramIndex = GetBlockParamIndex(texelX, texelY, numBlocksU);

                const BC6Parameters *texture = feature.grid[mip];
                const BC6Parameters &blockParameters = texture[paramIndex];

                int mask = gBC6HPartitionSets[blockParameters.partition];
                int partition = (mask >> pixelIndex) & 0x1;
                int endpointBase = 2 * partition;
                float3 endpoint0 = blockParameters.endpoints[endpointBase + 0];
                float3 endpoint1 = blockParameters.endpoints[endpointBase + 1];
                const int endpointBits = gBC6HPartitionNumBits[blockParameters.mode];
                float pixelIndexOnSegment = blockParameters.pixelIndices[pixelIndex];

                float3 ebar0, ebar1, y;
                DecompressBC6IntermediateValues(
                    endpoint0, endpoint1, endpointBits, pixelIndexOnSegment, ebar0, ebar1, y);

                float3 featureGrad =
                    weight * make_float3(float(inputGradient[3 * featureIndex]),
                                         float(inputGradient[3 * featureIndex + 1]),
                                         float(inputGradient[3 * featureIndex + 2]));

                // Backpropagate through BC6
                const int q = 3;
                int3 hy = make_int3(Max(Floor((y - 1.f) / 1024.f) - 1.f, 0.f);
                // float3 dw_dy = make_float3(1 << (hy - 24));
                const float3 dl_dy = Ldexp(featureGrad, hy - 24);

                float alpha = ldexpf(pixelIndexOnSegment, q);
                float3 dy_dx = Ldexp(ebar1 - ebar0, q);

                const float a = 31.f / 64;
                float deibar_dei = ldexpf(a, 16 - endpointBits);

                float3 dl_dx = dl_dy * dy_dx;
                float3 dl_de0 = (1.f - alpha) * deibar_dei * dl_dy;
                float3 dl_de1 = alpha * deibar_dei * dl_dy;
            }
        }
    }
} // namespace neural_textures

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
    (void)threadID;
    RNG rng;

    // for (int iter = 0; iter < gNumIters; iter++)
    {
        float u = rng.UniformFloat();
        float v = rng.UniformFloat();
        float s = rng.UniformFloat();

        half sampledFeatures[gNumFeatures * 3];
        SampleFourFeaturesTrilinear(params, make_float3(u, v, s), sampledFeatures);

        // TODO: feed the four sampled feature vectors into the MLP, compare against the
        // filtered reference material sample, and backpropagate into network + BC6 params.

        // TODO: sample the reference textures
        float expected[NT_OUTPUT_SIZE] = {};

        tcnn::hvec<NT_OUTPUT_SIZE> outputVec =
            ForwardPass(sampledFeatures, params.networkWeights[0], params.networkWeights[1]);

        // Calculate MSE loss
        float mse = 0.f;
        tcnn::hvec<NT_OUTPUT_SIZE> lossGradient;
        for (int i = 0; i < NT_OUTPUT_SIZE; i++)
        {
            float outputValue = __half2float(outputVec[i]);
            float error = outputValue - expected[i];
            mse += error * error;

            float loss = 2 * error / float(params.numSamples);
            lossGradient[i] = __float2half(loss);
        }

        // BackwardPass(lossGradient, params.networkWeights[0], params.networkWeights[1]);

        // gradient of loss w.r.t 12 inputs
    }
}

NT_DEVICE void OptimizePass(const KernelParams params) {}

} // namespace neural_textures
