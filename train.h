#pragma once
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace neural_textures
{

#define NT_INPUT_SIZE 12
#define NT_HIDDEN_LAYER_SIZE 16
#define NT_OUTPUT_SIZE 16
#define NT_MLP_HIDDEN_LAYER0_SIZE 32
#define NT_MLP_HIDDEN_LAYER1_SIZE 32

#define half __half

#define NT_NUM_NETWORK_LAYERS 3
#define NT_NUM_BC6_PIXELS_PER_BLOCK 16
#define NT_NUM_FEATURES 4
#define NT_MAX_REFERENCE_TEXTURES 16

enum class TrainingKernelType
{
    UNCONSTRAINED,
    BLOCK_FEATURES,
    FINALIZE,
};

struct AdamConstants
{
    float beta1;
    float beta2;
    float learningRate;
    float epsilon;
};

struct BC6Parameters
{
    float3 endpoints[4];
    float pixelIndices[NT_NUM_BC6_PIXELS_PER_BLOCK]; // [0, 1]
    int partition;
    int mode;
};

struct BC6ParameterGradients
{
    float3 endpoints[4];
    float pixelIndices[NT_NUM_BC6_PIXELS_PER_BLOCK];
};

struct Feature
{
    BC6Parameters **grid;
    BC6ParameterGradients **gradients;
    BC6ParameterGradients **moment1;
    BC6ParameterGradients **moment2;

    // Initial unconstrained features
    float3 **unconstrainedGrid;
    float3 **unconstrainedGradients;
    float3 **unconstrainedMoment1;
    float3 **unconstrainedMoment2;

    int width; // in texels
    int height;
    int numMips;
};

struct ReferenceTexture
{
    cudaTextureObject_t texture = 0;
    int width = 0;
    int height = 0;
    int numChannels = 0;
    int numMipLevels = 0;
    float lossWeight = 1.0f;
};

constexpr int AlignUpConstexpr(int value, int alignment)
{
    return ((value + alignment - 1) / alignment) * alignment;
}

static constexpr int kNetworkLayerInputSizes[NT_NUM_NETWORK_LAYERS] = {
    NT_INPUT_SIZE,
    NT_MLP_HIDDEN_LAYER0_SIZE,
    NT_MLP_HIDDEN_LAYER1_SIZE,
};

static constexpr int kNetworkLayerOutputSizes[NT_NUM_NETWORK_LAYERS] = {
    NT_MLP_HIDDEN_LAYER0_SIZE,
    NT_MLP_HIDDEN_LAYER1_SIZE,
    NT_OUTPUT_SIZE,
};

static constexpr int kNetworkLayerPaddedInputSizes[NT_NUM_NETWORK_LAYERS] = {
    AlignUpConstexpr(kNetworkLayerInputSizes[0], 16),
    AlignUpConstexpr(kNetworkLayerInputSizes[1], 16),
    AlignUpConstexpr(kNetworkLayerInputSizes[2], 16),
};

static constexpr int kNetworkLayerPaddedOutputSizes[NT_NUM_NETWORK_LAYERS] = {
    AlignUpConstexpr(kNetworkLayerOutputSizes[0], 16),
    AlignUpConstexpr(kNetworkLayerOutputSizes[1], 16),
    AlignUpConstexpr(kNetworkLayerOutputSizes[2], 16),
};

static constexpr int kNetworkLayerWeightCounts[NT_NUM_NETWORK_LAYERS] = {
    kNetworkLayerPaddedInputSizes[0] * kNetworkLayerPaddedOutputSizes[0],
    kNetworkLayerPaddedInputSizes[1] * kNetworkLayerPaddedOutputSizes[1],
    kNetworkLayerPaddedInputSizes[2] * kNetworkLayerPaddedOutputSizes[2],
};

static constexpr int kNetworkLayerBiasCounts[NT_NUM_NETWORK_LAYERS] = {
    kNetworkLayerOutputSizes[0],
    kNetworkLayerOutputSizes[1],
    kNetworkLayerOutputSizes[2],
};

struct KernelParams
{
    // params[mip][v * numU + u]
    Feature features[NT_NUM_FEATURES];

    // NOTE: padded to 16x16
    half *networkWeights[NT_NUM_NETWORK_LAYERS];
    float *networkWeightGradients[NT_NUM_NETWORK_LAYERS];
    float *networkWeightMoment1[NT_NUM_NETWORK_LAYERS];
    float *networkWeightMoment2[NT_NUM_NETWORK_LAYERS];

    half *networkBiases[NT_NUM_NETWORK_LAYERS];
    float *networkBiasGradients[NT_NUM_NETWORK_LAYERS];
    float *networkBiasMoment1[NT_NUM_NETWORK_LAYERS];
    float *networkBiasMoment2[NT_NUM_NETWORK_LAYERS];

    AdamConstants networkAdam;
    AdamConstants featureAdam;

    ReferenceTexture referenceTextures[NT_MAX_REFERENCE_TEXTURES];
    int numReferenceTextures = 0;
    int usedOutputChannels = NT_OUTPUT_SIZE;

    float *inferenceOutput = nullptr;
    float *inferenceMseAccum = nullptr;

    int imageWidth;
    int imageHeight;
    int numMips;
    int numBlocksU;
    int numBlocksV;

    int texelOffsetX;
    int texelOffsetY;

    int numSamples;
    int step = 0;
};
void InvokeOptimizeNetwork(KernelParams params);
void InvokeOptimizeFeatures(KernelParams params, TrainingKernelType type);
void InvokeTraining(KernelParams params, TrainingKernelType type);
void InvokeInference(KernelParams params);
} // namespace neural_textures
