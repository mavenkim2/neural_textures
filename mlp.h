#pragma once

#include <cuda_fp16.h>
#include <tiny-cuda-nn/mma.h>

#if defined(__CUDACC__)
#define NT_DEVICE __device__
#endif

#define WARP_SIZE 32
#define NT_INPUT_SIZE 12
#define NT_HIDDEN_LAYER_SIZE 16
#define NT_OUTPUT_SIZE 8

#define half __half

namespace neural_textures
{

NT_DEVICE inline tcnn::hvec<NT_OUTPUT_SIZE>
ForwardPass(const half *sampledFeatures,
            const half *weightsHidden0,
            const half *weightsHidden1,
            tcnn::hvec<NT_HIDDEN_LAYER_SIZE> *activatedHiddenLayer0 = 0)
{
    tcnn::hvec<NT_INPUT_SIZE> inputsVector0(sampledFeatures);
    tcnn::mma_mat<WARP_SIZE, 16, tcnn::RM> inputsMatrix0(inputsVector0);

    tcnn::mma_mat<16, 16, tcnn::CM> weightsHiddenLayer0 =
        tcnn::mma_mat<16, 16, tcnn::CM>::from_linear_memory(weightsHidden0);

    auto outputHiddenLayer0 = inputsMatrix0 * weightsHiddenLayer0;
    outputHiddenLayer0.activate<tcnn::Activation::ReLU>();

    if (activatedHiddenLayer0)
    {
        *activatedHiddenLayer0 = outputHiddenLayer0.vec<NT_HIDDEN_LAYER_SIZE>();
    }

    tcnn::mma_mat<16, 16, tcnn::CM> weightsHiddenLayer1 =
        tcnn::mma_mat<16, 16, tcnn::CM>::from_linear_memory(weightsHidden1);

    auto finalOutput = outputHiddenLayer0 * weightsHiddenLayer1;
    // finalOutput.activate<tcnn::Activation::ReLU>();

    return finalOutput.vec<NT_OUTPUT_SIZE>();
}

NT_DEVICE inline void BackwardPass(const tcnn::hvec<NT_OUTPUT_SIZE> &lossGradientVector,
                                   const tcnn::hvec<NT_HIDDEN_LAYER_SIZE> &activatedHiddenLayer0,
                                   const tcnn::hvec<NT_INPUT_SIZE> &networkInput,
                                   const half *weightsHidden0,
                                   const half *weightsHidden1)
{
    tcnn::mma_vec<16> lossGradientMatrix(lossGradientVector); // 32x8
    tcnn::mma_mat<16, 16, tcnn::CM> weightsOutput =
        tcnn::mma_mat<16, 16, tcnn::CM>::from_linear_memory(weightsHidden1);    // 16x8
    auto hiddenGradientMatrix = lossGradientMatrix * weightsOutput.transpose(); // 32x16

    tcnn::mma_vec<NT_HIDDEN_LAYER_SIZE> outputLayerInput(activatedHiddenLayer0); // 32x16
    auto outputWeightGradientMatrix =
        tcnn::outer_product(outputLayerInput, lossGradientMatrix); // 16x8

    tcnn::mma_mat<16, 16, tcnn::CM> weightsHiddenLayer0 =
        tcnn::mma_mat<16, 16, tcnn::CM>::from_linear_memory(weightsHidden0);           // 12x16
    auto inputGradientMatrix = hiddenGradientMatrix * weightsHiddenLayer0.transpose(); // 32x12

    tcnn::mma_vec<16> networkInputMatrix(networkInput); // 32x12
    auto hiddenWeightGradientMatrix =
        tcnn::outer_product(networkInputMatrix, hiddenGradientMatrix); // 12x16
}

} // namespace neural_textures
