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
ForwardPass(const half *sampledFeatures, const half *weightsHidden0, const half *weightsHidden1)
{
    tcnn::hvec<NT_INPUT_SIZE> inputsVector0(sampledFeatures);
    tcnn::mma_mat<WARP_SIZE, 16, tcnn::RM> inputsMatrix0(inputsVector0);

    tcnn::mma_mat<16, 16, tcnn::CM> weightsHiddenLayer0 =
        tcnn::mma_mat<16, 16, tcnn::CM>::from_linear_memory(weightsHidden0);

    auto outputHiddenLayer0 = inputsMatrix0 * weightsHiddenLayer0;
    outputHiddenLayer0.activate<tcnn::Activation::ReLU>();

    tcnn::mma_mat<16, 16, tcnn::CM> weightsHiddenLayer1 =
        tcnn::mma_mat<16, 16, tcnn::CM>::from_linear_memory(weightsHidden1);

    auto finalOutput = outputHiddenLayer0 * weightsHiddenLayer1;
    finalOutput.activate<tcnn::Activation::ReLU>();

    return finalOutput.vec<NT_OUTPUT_SIZE>();
}

} // namespace neural_textures
