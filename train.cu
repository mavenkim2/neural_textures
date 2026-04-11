#include <cmath>
#include <cstdint>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "filter/catmull_rom.h"
#include "mlp.h"

#include "train.h"
#include "util/atomic.h"
#include "util/float2.h"
#include "util/float3.h"
#include "util/float4.h"

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
    uint32_t state = 1u;

    NT_DEVICE explicit RNG(uint32_t seed = 1u) : state(seed ? seed : 1u) {}

    NT_DEVICE uint32_t NextUInt()
    {
        // xorshift32
        uint32_t x = state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        state = x;
        return x;
    }

    NT_DEVICE float UniformFloat()
    {
        return (float)(NextUInt() & 0x00FFFFFFu) * (1.0f / 16777216.0f);
    }
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

template <typename T>
NT_DEVICE inline T SampleReferenceTextureTrilinear(const ReferenceTexture &texture,
                                                   const float3 &uvs)
{
    // TODO: will the 1x1 final mip be correctly handled by this?
    const int maxMip = texture.numMipLevels > 0 ? texture.numMipLevels - 1 : 0;
    const float mipCoord = Clamp(uvs.z, 0.f, (float)maxMip);
    const int mip0 = min((int)floorf(mipCoord), maxMip);
    const int mip1 = min(mip0 + 1, maxMip);
    const float mipFrac = mip1 > mip0 ? (mipCoord - (float)mip0) : 0.f;

    const float2 uv = make_float2(uvs.x, uvs.y);
    const float2 texSize0 = make_float2((float)GetMipDimension(texture.width, mip0),
                                        (float)GetMipDimension(texture.height, mip0));
    T sample0 = SampleTextureCatmullRomLod<T>(texture.texture, uv, texSize0, (float)mip0);

    if (mip1 == mip0)
    {
        return sample0;
    }

    const float2 texSize1 = make_float2((float)GetMipDimension(texture.width, mip1),
                                        (float)GetMipDimension(texture.height, mip1));
    T sample1 = SampleTextureCatmullRomLod<T>(texture.texture, uv, texSize1, (float)mip1);
    return sample0 * (1.f - mipFrac) + sample1 * mipFrac;
}

NT_DEVICE inline void
InitializeExpectedValues(const KernelParams &params, const float3 &uvs, float *expected)
{
    for (int i = 0; i < NT_OUTPUT_SIZE; ++i)
    {
        expected[i] = 0.f;
    }

    int outputChannel = 0;
    const int textureCount = params.numReferenceTextures < NT_MAX_REFERENCE_TEXTURES
                                 ? params.numReferenceTextures
                                 : NT_MAX_REFERENCE_TEXTURES;

    for (int textureIndex = 0; textureIndex < textureCount && outputChannel < NT_OUTPUT_SIZE;
         ++textureIndex)
    {
        const ReferenceTexture &texture = params.referenceTextures[textureIndex];
        if (texture.texture == 0 || texture.numChannels <= 0)
        {
            continue;
        }

        if (texture.numChannels == 1)
        {
            expected[outputChannel++] = SampleReferenceTextureTrilinear<float>(texture, uvs);
            continue;
        }

        const float4 sample = SampleReferenceTextureTrilinear<float4>(texture, uvs);
        const float channels[4] = {sample.x, sample.y, sample.z, sample.w};
        const int channelCount = texture.numChannels < 4 ? texture.numChannels : 4;
        for (int channelIndex = 0; channelIndex < channelCount && outputChannel < NT_OUTPUT_SIZE;
             ++channelIndex)
        {
            expected[outputChannel++] = channels[channelIndex];
        }
    }
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

NT_HOST_DEVICE inline int GetMipDimension(int size, int mip)
{
    int dimension = size >> mip;
    return dimension > 0 ? dimension : 1;
}

NT_HOST_DEVICE inline int GetBlockCountForDimension(int size)
{
    int numBlocks = size >> 2;
    return numBlocks > 0 ? numBlocks : 1;
}

NT_HOST_DEVICE inline int GetNumBlocksAtMip(const Feature &feature, int mip)
{
    int width = GetMipDimension(feature.width, mip);
    int height = GetMipDimension(feature.height, mip);
    return GetBlockCountForDimension(width) * GetBlockCountForDimension(height);
}

NT_HOST_DEVICE inline int GetNumTexelsAtMip(const Feature &feature, int mip)
{
    int width = GetMipDimension(feature.width, mip);
    int height = GetMipDimension(feature.height, mip);
    return width * height;
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
SampleUnconstrainedFeaturesTrilinear(const KernelParams &params, float3 uvs, half *outFeatures)
{
    int mipBase = (int)floorf(uvs.z);
    float mipFrac = uvs.z - floorf(uvs.z);

    for (int featureIndex = 0; featureIndex < NT_NUM_FEATURES * 3; ++featureIndex)
    {
        outFeatures[featureIndex] = 0;
    }

    for (int featureIndex = 0; featureIndex < NT_NUM_FEATURES; ++featureIndex)
    {
        const Feature &feature = params.features[featureIndex];
        const int maxMip = feature.numMips > 0 ? feature.numMips - 1 : 0;
        const int maxMipIndex = mipFrac > 0.f ? 1 : 0;
        for (int mipIndex = 0; mipIndex <= maxMipIndex; mipIndex++)
        {
            int mip = min(mipBase + mipIndex, maxMip);
            const float3 *texture = feature.unconstrainedGrid[mip];

            int width = max(feature.width >> mip, 1);
            int height = max(feature.height >> mip, 1);

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

                float3 feature = weight * texture[width * texelY + texelX];

                outFeatures[3 * featureIndex] += __float2half(feature.x);
                outFeatures[3 * featureIndex + 1] += __float2half(feature.y);
                outFeatures[3 * featureIndex + 2] += __float2half(feature.z);
            }
        }
    }
}

NT_DEVICE inline void
SampleFourFeaturesTrilinear(const KernelParams &params, float3 uvs, half *outFeatures)
{
    int mipBase = (int)floorf(uvs.z);
    float mipFrac = uvs.z - floorf(uvs.z);

    for (int featureIndex = 0; featureIndex < NT_NUM_FEATURES * 3; ++featureIndex)
    {
        outFeatures[featureIndex] = 0;
    }

    // TODO: consider reordering loads/global memory accesses

    for (int featureIndex = 0; featureIndex < NT_NUM_FEATURES; ++featureIndex)
    {
        const Feature &feature = params.features[featureIndex];
        const int maxMip = feature.numMips > 0 ? feature.numMips - 1 : 0;
        const int maxMipIndex = mipFrac > 0.f ? 1 : 0;
        for (int mipIndex = 0; mipIndex <= maxMipIndex; mipIndex++)
        {
            int mip = min(mipBase + mipIndex, maxMip);
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

NT_DEVICE inline void BackwardFeaturePass(KernelParams &params,
                                          float3 uvs,
                                          const tcnn::hvec<NT_INPUT_SIZE> &inputGradient)
{
    int mipBase = (int)floorf(uvs.z);
    float mipFrac = uvs.z - floorf(uvs.z);

    // TODO: consider reordering loads/global memory accesses
    for (int featureIndex = 0; featureIndex < NT_NUM_FEATURES; ++featureIndex)
    {
        Feature &feature = params.features[featureIndex];
        const int maxMip = feature.numMips > 0 ? feature.numMips - 1 : 0;
        const int maxMipIndex = mipFrac > 0.f ? 1 : 0;
        for (int mipIndex = 0; mipIndex <= maxMipIndex; mipIndex++)
        {
            int mip = min(mipBase + mipIndex, maxMip);
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

                const int pixelIndex = GetBlockPixelIndex(texelX, texelY);
                const int paramIndex = GetBlockParamIndex(texelX, texelY, numBlocksU);

                const BC6Parameters *texture = feature.grid[mip];
                BC6ParameterGradients &outGradients = feature.gradients[mip][paramIndex];
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
                int3 hy = make_int3(Max(Floor((y - 1.f) / 1024.f) - 1.f, 0.f));
                // float3 dw_dy = make_float3(1 << (hy - 24));
                const float3 dl_dy = Ldexp(featureGrad, hy - 24);

                float alpha = ldexpf(pixelIndexOnSegment, q);
                float3 dy_dx = Ldexp(ebar1 - ebar0, q);

                const float a = 31.f / 64;
                float deibar_dei = ldexpf(a, 16 - endpointBits);

                float dl_dx = Dot(dl_dy, dy_dx);
                float3 dl_de0 = (1.f - alpha) * deibar_dei * dl_dy;
                float3 dl_de1 = alpha * deibar_dei * dl_dy;

                atomicAdd(&outGradients.endpoints[endpointBase + 0], dl_de0);
                atomicAdd(&outGradients.endpoints[endpointBase + 1], dl_de1);
                atomicAdd(&outGradients.pixelIndices[pixelIndex], dl_dx);
            }
        }
    }
}

NT_DEVICE inline void BackwardUnconstrainedFeaturePass(
    KernelParams &params, float3 uvs, const tcnn::hvec<NT_INPUT_SIZE> &inputGradient)
{
    int mipBase = (int)floorf(uvs.z);
    float mipFrac = uvs.z - floorf(uvs.z);

    // TODO: consider reordering loads/global memory accesses
    for (int featureIndex = 0; featureIndex < NT_NUM_FEATURES; ++featureIndex)
    {
        Feature &feature = params.features[featureIndex];
        const int maxMip = feature.numMips > 0 ? feature.numMips - 1 : 0;
        const int maxMipIndex = mipFrac > 0.f ? 1 : 0;
        for (int mipIndex = 0; mipIndex <= maxMipIndex; mipIndex++)
        {
            int mip = min(mipBase + mipIndex, maxMip);
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

                const int texelIndex = texelY * width + texelX;
                float3 *texture = feature.unconstrainedGradients[mip];

                float3 featureGrad =
                    weight * make_float3(float(inputGradient[3 * featureIndex]),
                                         float(inputGradient[3 * featureIndex + 1]),
                                         float(inputGradient[3 * featureIndex + 2]));
                atomicAdd(texture + texelIndex, featureGrad);
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

NT_DEVICE inline void AdamUpdateParameter(float &moment1,
                                          float &moment2,
                                          float gradient,
                                          const AdamConstants &adam,
                                          int step,
                                          half &weight)
{
    const int currentStep = step > 0 ? step : 1;
    const float beta1 = adam.beta1;
    const float beta2 = adam.beta2;
    const float biasCorrection1 = 1.f - powf(beta1, (float)currentStep);
    const float biasCorrection2 = 1.f - powf(beta2, (float)currentStep);

    moment1 = beta1 * moment1 + (1.f - beta1) * gradient;
    moment2 = beta2 * moment2 + (1.f - beta2) * gradient * gradient;

    const float mhat = moment1 / biasCorrection1;
    const float vhat = moment2 / biasCorrection2;
    const float weightValue = __half2float(weight);
    const float newWeight = weightValue - adam.learningRate * mhat / (sqrtf(vhat) + adam.epsilon);

    weight = __float2half(newWeight);
}

NT_DEVICE inline void AdamUpdateFloatParameter(float &moment1,
                                               float &moment2,
                                               float gradient,
                                               const AdamConstants &adam,
                                               int step,
                                               float &weight)
{
    const int currentStep = step > 0 ? step : 1;
    const float beta1 = adam.beta1;
    const float beta2 = adam.beta2;
    const float biasCorrection1 = 1.f - powf(beta1, (float)currentStep);
    const float biasCorrection2 = 1.f - powf(beta2, (float)currentStep);

    moment1 = beta1 * moment1 + (1.f - beta1) * gradient;
    moment2 = beta2 * moment2 + (1.f - beta2) * gradient * gradient;

    const float mhat = moment1 / biasCorrection1;
    const float vhat = moment2 / biasCorrection2;
    weight -= adam.learningRate * mhat / (sqrtf(vhat) + adam.epsilon);
}

NT_DEVICE inline void OptimizeParameterBuffer(half *weights,
                                              float *gradients,
                                              float *moment1,
                                              float *moment2,
                                              int count,
                                              const AdamConstants &adam,
                                              int step)
{
    const int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    const int threadCount = gridDim.x * blockDim.x;

    for (int i = threadId; i < count; i += threadCount)
    {
        const float gradient = gradients[i];
        if (gradient == 0.f)
        {
            continue;
        }

        AdamUpdateParameter(moment1[i], moment2[i], gradient, adam, step, weights[i]);
        gradients[i] = 0.f;
    }
}

NT_DEVICE inline void OptimizePaddedParameterMatrix(half *weights,
                                                    float *gradients,
                                                    float *moment1,
                                                    float *moment2,
                                                    int activeRows,
                                                    int activeCols,
                                                    int paddedRows,
                                                    const AdamConstants &adam,
                                                    int step)
{
    const int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    const int threadCount = gridDim.x * blockDim.x;
    const int activeCount = activeRows * activeCols;

    for (int activeIndex = threadId; activeIndex < activeCount; activeIndex += threadCount)
    {
        const int row = activeIndex % activeRows;
        const int col = activeIndex / activeRows;
        const int linearIndex = col * paddedRows + row;
        const float gradient = gradients[linearIndex];
        if (gradient == 0.f)
        {
            continue;
        }

        AdamUpdateParameter(moment1[linearIndex],
                            moment2[linearIndex],
                            gradient,
                            adam,
                            step,
                            weights[linearIndex]);
        gradients[linearIndex] = 0.f;
    }
}

NT_DEVICE inline void OptimizeFeatureScalar(float &weight,
                                            float &gradient,
                                            float &moment1,
                                            float &moment2,
                                            const AdamConstants &adam,
                                            int step)
{
    if (gradient == 0.f)
    {
        return;
    }

    AdamUpdateFloatParameter(moment1, moment2, gradient, adam, step, weight);
    gradient = 0.f;
}

NT_DEVICE inline void OptimizeFeatureBlock(BC6Parameters &parameters,
                                           BC6ParameterGradients &gradients,
                                           BC6ParameterGradients &moment1,
                                           BC6ParameterGradients &moment2,
                                           const AdamConstants &adam,
                                           int step)
{
    for (int endpointIndex = 0; endpointIndex < 4; ++endpointIndex)
    {
        OptimizeFeatureScalar(parameters.endpoints[endpointIndex].x,
                              gradients.endpoints[endpointIndex].x,
                              moment1.endpoints[endpointIndex].x,
                              moment2.endpoints[endpointIndex].x,
                              adam,
                              step);
        OptimizeFeatureScalar(parameters.endpoints[endpointIndex].y,
                              gradients.endpoints[endpointIndex].y,
                              moment1.endpoints[endpointIndex].y,
                              moment2.endpoints[endpointIndex].y,
                              adam,
                              step);
        OptimizeFeatureScalar(parameters.endpoints[endpointIndex].z,
                              gradients.endpoints[endpointIndex].z,
                              moment1.endpoints[endpointIndex].z,
                              moment2.endpoints[endpointIndex].z,
                              adam,
                              step);
    }

    for (int pixelIndex = 0; pixelIndex < NT_NUM_BC6_PIXELS_PER_BLOCK; ++pixelIndex)
    {
        OptimizeFeatureScalar(parameters.pixelIndices[pixelIndex],
                              gradients.pixelIndices[pixelIndex],
                              moment1.pixelIndices[pixelIndex],
                              moment2.pixelIndices[pixelIndex],
                              adam,
                              step);
    }
}

NT_DEVICE inline void OptimizeNetworkPass(KernelParams params)
{
    constexpr int paddedWeightRows = NT_HIDDEN_LAYER_SIZE;
    constexpr int weightRows[NT_NUM_NETWORK_LAYERS] = {
        NT_INPUT_SIZE,
        NT_HIDDEN_LAYER_SIZE,
    };
    constexpr int weightCols[NT_NUM_NETWORK_LAYERS] = {
        NT_HIDDEN_LAYER_SIZE,
        NT_OUTPUT_SIZE,
    };
    constexpr int biasCounts[NT_NUM_NETWORK_LAYERS] = {
        NT_HIDDEN_LAYER_SIZE,
        NT_OUTPUT_SIZE,
    };

    for (int layer = 0; layer < NT_NUM_NETWORK_LAYERS; ++layer)
    {
        OptimizePaddedParameterMatrix(params.networkWeights[layer],
                                      params.networkWeightGradients[layer],
                                      params.networkWeightMoment1[layer],
                                      params.networkWeightMoment2[layer],
                                      weightRows[layer],
                                      weightCols[layer],
                                      paddedWeightRows,
                                      params.networkAdam,
                                      params.step);

        OptimizeParameterBuffer(params.networkBiases[layer],
                                params.networkBiasGradients[layer],
                                params.networkBiasMoment1[layer],
                                params.networkBiasMoment2[layer],
                                biasCounts[layer],
                                params.networkAdam,
                                params.step);
    }
}

NT_DEVICE inline void OptimizeFeaturePass(KernelParams params)
{
    const int linearThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const int threadCount = gridDim.x * blockDim.x;
    int baseBlockIndex = 0;

    for (int featureIndex = 0; featureIndex < NT_NUM_FEATURES; ++featureIndex)
    {
        Feature &feature = params.features[featureIndex];
        for (int mip = 0; mip < feature.numMips; ++mip)
        {
            const int numBlocks = GetNumBlocksAtMip(feature, mip);
            const int segmentEnd = baseBlockIndex + numBlocks;

            for (int linearBlockIndex = linearThreadIndex; linearBlockIndex < segmentEnd;
                 linearBlockIndex += threadCount)
            {
                if (linearBlockIndex < baseBlockIndex)
                {
                    continue;
                }

                const int paramIndex = linearBlockIndex - baseBlockIndex;
                OptimizeFeatureBlock(feature.grid[mip][paramIndex],
                                     feature.gradients[mip][paramIndex],
                                     feature.moment1[mip][paramIndex],
                                     feature.moment2[mip][paramIndex],
                                     params.featureAdam,
                                     params.step);
            }

            baseBlockIndex = segmentEnd;
        }
    }
}

NT_DEVICE inline void OptimizeUnconstrainedFeaturePass(KernelParams params)
{
    const int linearThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const int threadCount = gridDim.x * blockDim.x;
    int baseTexelIndex = 0;

    for (int featureIndex = 0; featureIndex < NT_NUM_FEATURES; ++featureIndex)
    {
        Feature &feature = params.features[featureIndex];
        for (int mip = 0; mip < feature.numMips; ++mip)
        {
            const int numTexels = GetNumTexelsAtMip(feature, mip);
            const int segmentEnd = baseTexelIndex + numTexels;

            for (int linearTexelIndex = linearThreadIndex; linearTexelIndex < segmentEnd;
                 linearTexelIndex += threadCount)
            {
                if (linearTexelIndex < baseTexelIndex)
                {
                    continue;
                }

                const int texelIndex = linearTexelIndex - baseTexelIndex;
                float3 &weight = feature.unconstrainedGrid[mip][texelIndex];
                float3 &gradient = feature.unconstrainedGradients[mip][texelIndex];
                float3 &moment1 = feature.unconstrainedMoment1[mip][texelIndex];
                float3 &moment2 = feature.unconstrainedMoment2[mip][texelIndex];

                OptimizeFeatureScalar(
                    weight.x, gradient.x, moment1.x, moment2.x, params.featureAdam, params.step);
                OptimizeFeatureScalar(
                    weight.y, gradient.y, moment1.y, moment2.y, params.featureAdam, params.step);
                OptimizeFeatureScalar(
                    weight.z, gradient.z, moment1.z, moment2.z, params.featureAdam, params.step);
            }

            baseTexelIndex = segmentEnd;
        }
    }
}

template <TrainingKernelType type>
NT_DEVICE inline void OptimizeFeaturesPass(KernelParams params)
{
    if constexpr (type == TrainingKernelType::UNCONSTRAINED)
    {
        OptimizeUnconstrainedFeaturePass(params);
    }
    else
    {
        OptimizeFeaturePass(params);
    }
}

template <uint32_t numThreads, TrainingKernelType type>
NT_DEVICE void TrainLoopPass(KernelParams params)
{
    const uint32_t sampleIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (sampleIndex >= (uint32_t)params.numSamples)
    {
        return;
    }

    constexpr int kTrainingRegionSize = 512;
    const uint32_t threadX = sampleIndex % kTrainingRegionSize;
    const uint32_t threadY = sampleIndex / kTrainingRegionSize;
    const uint32_t sampleX = params.texelOffsetX + threadX;
    const uint32_t sampleY = params.texelOffsetY + threadY;

    RNG rng(sampleIndex ^ ((uint32_t)params.step * 747796405u + 2891336453u));

    {
        float jitterX = rng.UniformFloat() - 0.5f;
        float jitterY = rng.UniformFloat() - 0.5f;
        float samplePosX =
            Clamp((float)sampleX + 0.5f + jitterX, 0.5f, (float)params.imageWidth - 0.5f);
        float samplePosY =
            Clamp((float)sampleY + 0.5f + jitterY, 0.5f, (float)params.imageHeight - 0.5f);
        float u = samplePosX / (float)params.imageWidth;
        float v = samplePosY / (float)params.imageHeight;
        float3 uvs = make_float3(u, v, 0.f);

        half sampledFeatures[NT_NUM_FEATURES * 3];
        if constexpr (type == TrainingKernelType::UNCONSTRAINED)
        {
            SampleUnconstrainedFeaturesTrilinear(params, uvs, sampledFeatures);
        }
        else
        {
            SampleFourFeaturesTrilinear(params, uvs, sampledFeatures);
        }

        // TODO: feed the four sampled feature vectors into the MLP, compare against the
        // filtered reference material sample, and backpropagate into network + BC6 params.

        float expected[NT_OUTPUT_SIZE] = {};
        InitializeExpectedValues(params, uvs, expected);

        tcnn::hvec<NT_HIDDEN_LAYER_SIZE> activatedHiddenLayer0;
        tcnn::hvec<NT_OUTPUT_SIZE> outputVec = ForwardPass(sampledFeatures,
                                                           params.networkWeights[0],
                                                           params.networkWeights[1],
                                                           params.networkBiases[0],
                                                           params.networkBiases[1],
                                                           &activatedHiddenLayer0);

        // Calculate MSE loss
        float mse = 0.f;
        tcnn::hvec<NT_OUTPUT_SIZE> lossGradient;
        for (int i = 0; i < NT_OUTPUT_SIZE; i++)
        {
            float outputValue = __half2float(outputVec[i]);
            float error = outputValue - expected[i];
            mse += error * error;

            float loss = 2.f * error / float(params.numSamples);
            lossGradient[i] = __float2half(loss);
        }

        tcnn::hvec<NT_INPUT_SIZE> inputsVector(sampledFeatures);
        tcnn::hvec<NT_INPUT_SIZE> inputGradient =
            BackwardPass<numThreads>(lossGradient,
                                     activatedHiddenLayer0,
                                     inputsVector,
                                     params.networkWeights[0],
                                     params.networkWeights[1],
                                     params.networkWeightGradients[0],
                                     params.networkWeightGradients[1],
                                     params.networkBiasGradients[0],
                                     params.networkBiasGradients[1]);

        if constexpr (type == TrainingKernelType::UNCONSTRAINED)
        {
            BackwardUnconstrainedFeaturePass(params, uvs, inputGradient);
        }
        else if constexpr (type == TrainingKernelType::BLOCK_FEATURES)
        {
            BackwardFeaturePass(params, uvs, inputGradient);
        }
    }
}

NT_DEVICE void Inference(KernelParams params)
{
    const uint32_t sampleIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t numInferenceSamples = (uint32_t)(params.imageWidth * params.imageHeight);
    if (sampleIndex >= numInferenceSamples)
    {
        return;
    }

    const uint32_t sampleX = sampleIndex % (uint32_t)params.imageWidth;
    const uint32_t sampleY = sampleIndex / (uint32_t)params.imageWidth;

    const float samplePosX = (float)sampleX + 0.5f;
    const float samplePosY = (float)sampleY + 0.5f;
    const float u = samplePosX / (float)params.imageWidth;
    const float v = samplePosY / (float)params.imageHeight;
    const float3 uvs = make_float3(u, v, 0.f);

    half sampledFeatures[NT_NUM_FEATURES * 3];
    SampleUnconstrainedFeaturesTrilinear(params, uvs, sampledFeatures);

    float expected[NT_OUTPUT_SIZE] = {};
    InitializeExpectedValues(params, uvs, expected);

    tcnn::hvec<NT_OUTPUT_SIZE> outputVec = ForwardPass(sampledFeatures,
                                                       params.networkWeights[0],
                                                       params.networkWeights[1],
                                                       params.networkBiases[0],
                                                       params.networkBiases[1]);

    float sampleSse = 0.f;
    const uint32_t outputBaseIndex = sampleIndex * NT_OUTPUT_SIZE;
    for (int i = 0; i < NT_OUTPUT_SIZE; ++i)
    {
        const float outputValue = __half2float(outputVec[i]);
        params.inferenceOutput[outputBaseIndex + (uint32_t)i] = outputValue;

        const float error = outputValue - expected[i];
        sampleSse += error * error;
    }

    atomicAdd(params.inferenceMseAccum, sampleSse);
}

__global__ void OptimizeNetwork(KernelParams params)
{
    OptimizeNetworkPass(params);
}

template <TrainingKernelType type>
__global__ void OptimizeFeatures(KernelParams params)
{
    OptimizeFeaturesPass<type>(params);
}

__global__ void RunInference(KernelParams params)
{
    Inference(params);
}

template <uint32_t numThreads, TrainingKernelType type>
__global__ void TrainLoop(KernelParams params)
{
    TrainLoopPass<numThreads, type>(params);
}

void InvokeOptimizeNetwork(KernelParams params)
{
    OptimizeNetwork<<<1, 32>>>(params);
}

void InvokeOptimizeFeatures(KernelParams params, TrainingKernelType type)
{
    int totalElements = 0;
    for (int featureIndex = 0; featureIndex < NT_NUM_FEATURES; ++featureIndex)
    {
        const Feature &feature = params.features[featureIndex];
        for (int mip = 0; mip < feature.numMips; ++mip)
        {
            if (type == TrainingKernelType::UNCONSTRAINED)
            {
                totalElements += GetNumTexelsAtMip(feature, mip);
            }
            else
            {
                totalElements += GetNumBlocksAtMip(feature, mip);
            }
        }
    }

    if (totalElements == 0)
    {
        return;
    }

    constexpr int kThreadsPerBlock = 128;
    const int numBlocks = (totalElements + kThreadsPerBlock - 1) / kThreadsPerBlock;
    if (type == TrainingKernelType::UNCONSTRAINED)
    {
        OptimizeFeatures<TrainingKernelType::UNCONSTRAINED>
            <<<numBlocks, kThreadsPerBlock>>>(params);
    }
    else
    {
        OptimizeFeatures<TrainingKernelType::BLOCK_FEATURES>
            <<<numBlocks, kThreadsPerBlock>>>(params);
    }
}

void InvokeTraining(KernelParams params, TrainingKernelType type)
{
    const int kThreadsPerBlock = 256;
    const int numSamples = params.numSamples > 0 ? params.numSamples : 512 * 512;
    const int numBlocks = (numSamples + kThreadsPerBlock - 1) / kThreadsPerBlock;
    if (type == TrainingKernelType::UNCONSTRAINED)
    {
        TrainLoop<kThreadsPerBlock, TrainingKernelType::UNCONSTRAINED>
            <<<numBlocks, kThreadsPerBlock>>>(params);
    }
    else if (type == TrainingKernelType::BLOCK_FEATURES)
    {
        TrainLoop<kThreadsPerBlock, TrainingKernelType::BLOCK_FEATURES>
            <<<numBlocks, kThreadsPerBlock>>>(params);
    }
    else
    {
        TrainLoop<kThreadsPerBlock, TrainingKernelType::FINALIZE>
            <<<numBlocks, kThreadsPerBlock>>>(params);
    }
}

void InvokeInference(KernelParams params)
{
    const int numSamples = params.imageWidth * params.imageHeight;
    if (numSamples <= 0 || params.inferenceOutput == nullptr ||
        params.inferenceMseAccum == nullptr)
    {
        return;
    }

    const int kThreadsPerBlock = 256;
    const int numBlocks = (numSamples + kThreadsPerBlock - 1) / kThreadsPerBlock;
    RunInference<<<numBlocks, kThreadsPerBlock>>>(params);
}

} // namespace neural_textures
