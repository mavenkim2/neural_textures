#include <cmath>
#include <cstdint>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "mlp.h"

#include "train.h"
#include "util/atomic.h"
#include "util/float2.h"
#include "util/float3.h"

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

    for (int featureIndex = 0; featureIndex < NT_NUM_FEATURES * 3; ++featureIndex)
    {
        outFeatures[featureIndex] = 0;
    }

    // TODO: consider reordering loads/global memory accesses

    for (int featureIndex = 0; featureIndex < NT_NUM_FEATURES; ++featureIndex)
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
        for (int mipIndex = 0; mipIndex <= 1; mipIndex++)
        {
            int mip = mipBase + mipIndex;
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
        for (int mipIndex = 0; mipIndex <= 1; mipIndex++)
        {
            int mip = mipBase + mipIndex;
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

                float3 *texture = feature.unconstrainedGrid[mip];

                float3 featureGrad =
                    weight * make_float3(float(inputGradient[3 * featureIndex]),
                                         float(inputGradient[3 * featureIndex + 1]),
                                         float(inputGradient[3 * featureIndex + 2]));
                atomicAdd(texture + paramIndex, featureGrad);
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
    constexpr int weightCounts[NT_NUM_NETWORK_LAYERS] = {
        NT_HIDDEN_LAYER_SIZE * NT_INPUT_SIZE,
        NT_HIDDEN_LAYER_SIZE * NT_OUTPUT_SIZE,
    };
    constexpr int biasCounts[NT_NUM_NETWORK_LAYERS] = {
        NT_HIDDEN_LAYER_SIZE,
        NT_OUTPUT_SIZE,
    };

    for (int layer = 0; layer < NT_NUM_NETWORK_LAYERS; ++layer)
    {
        OptimizeParameterBuffer(params.networkWeights[layer],
                                params.networkWeightGradients[layer],
                                params.networkWeightMoment1[layer],
                                params.networkWeightMoment2[layer],
                                weightCounts[layer],
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
        for (int mip = 0; mip < params.numMips; ++mip)
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

template <uint32_t numThreads, TrainingKernelType type>
NT_DEVICE void TrainLoopPass(KernelParams params)
{
    const uint32_t threadID = threadIdx.y * blockDim.x + threadIdx.x;
    (void)threadID;
    RNG rng;

    {
        float u = rng.UniformFloat();
        float v = rng.UniformFloat();
        float s = rng.UniformFloat();
        float3 uvs = make_float3(u, v, s);

        half sampledFeatures[NT_NUM_FEATURES * 3];
        SampleFourFeaturesTrilinear(params, uvs, sampledFeatures);

        // TODO: feed the four sampled feature vectors into the MLP, compare against the
        // filtered reference material sample, and backpropagate into network + BC6 params.

        // TODO: sample the reference textures
        float expected[NT_OUTPUT_SIZE] = {};

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

            float loss = 2 * error / float(params.numSamples);
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

__global__ void OptimizeNetwork(KernelParams params)
{
    OptimizeNetworkPass(params);
}

__global__ void OptimizeFeatures(KernelParams params)
{
    OptimizeFeaturePass(params);
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

void InvokeOptimizeFeatures(KernelParams params)
{
    int totalBlocks = 0;
    for (int featureIndex = 0; featureIndex < NT_NUM_FEATURES; ++featureIndex)
    {
        const Feature &feature = params.features[featureIndex];
        for (int mip = 0; mip < params.numMips; ++mip)
        {
            totalBlocks += GetNumBlocksAtMip(feature, mip);
        }
    }

    if (totalBlocks == 0)
    {
        return;
    }

    constexpr int kThreadsPerBlock = 128;
    const int numBlocks = (totalBlocks + kThreadsPerBlock - 1) / kThreadsPerBlock;
    OptimizeFeatures<<<numBlocks, kThreadsPerBlock>>>(params);
}

void InvokeTraining(KernelParams params, TrainingKernelType type)
{
    const int kThreadsPerBlock = 256;
    const int numSamples = 512 * 512;
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

} // namespace neural_textures
