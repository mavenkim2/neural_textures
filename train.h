#pragma once
#include <cstdint>
#include <cuda_fp16.h>

namespace neural_textures
{

#define NT_INPUT_SIZE 12
#define NT_HIDDEN_LAYER_SIZE 16
#define NT_OUTPUT_SIZE 8

#define half __half

#define NT_NUM_NETWORK_LAYERS 2
#define NT_NUM_BC6_PIXELS_PER_BLOCK 16
#define NT_NUM_FEATURES 4

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

    int imageWidth;
    int imageHeight;
    int numMips;
    int numBlocksU;
    int numBlocksV;

    int numSamples;
    int step = 0;
};
void InvokeOptimizeNetwork(KernelParams &params);
void InvokeOptimizeFeatures(KernelParams &params);
} // namespace neural_textures
