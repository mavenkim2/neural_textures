#pragma once

#include "train.h"
#include "util/common.h"
#include "util/cuda_util.h"
#include <cmath>
#include <tiny-cuda-nn/mma.h>

namespace neural_textures
{

enum class LayerActivation
{
    None,
    ReLU,
};

template <int inputSize, int outputSize, LayerActivation activation = LayerActivation::ReLU>
NT_DEVICE inline tcnn::mma_vec<AlignUp(outputSize, 16)>
ForwardPassGeneric(const tcnn::mma_vec<AlignUp(inputSize, 16)> &inputsMatrix,
                   const half *weightsLayer,
                   const half *biasesLayer)
{
    constexpr int weightsRows = AlignUp(inputSize, 16);
    constexpr int weightsColumns = AlignUp(outputSize, 16);

    using weightsMatrix = tcnn::mma_mat<weightsRows, weightsColumns, tcnn::CM>;

    weightsMatrix weights = weightsMatrix::from_linear_memory(weightsLayer);
    tcnn::hvec<outputSize> biasesVec(biasesLayer);
    tcnn::mma_vec<weightsColumns> biases(biasesVec);

    auto outputMatrix = madd(inputsMatrix, weights, biases);

    if constexpr (activation == LayerActivation::ReLU)
    {
        outputMatrix.template activate<tcnn::Activation::ReLU>();
    }
    return outputMatrix;
}

NT_DEVICE inline tcnn::hvec<NT_OUTPUT_SIZE>
ForwardPass12x16xOutput(const half *sampledFeatures,
                        const half *weightsLayer0,
                        const half *weightsLayer1,
                        const half *biasesLayer0,
                        const half *biasesLayer1,
                        tcnn::hvec<NT_HIDDEN_LAYER_SIZE> *activatedHiddenLayer0 = 0)
{
    tcnn::hvec<NT_INPUT_SIZE> inputsVector(sampledFeatures);
    tcnn::mma_vec<16> inputsMatrix(inputsVector);
    auto outputHiddenLayer = ForwardPassGeneric<NT_INPUT_SIZE, NT_HIDDEN_LAYER_SIZE>(
        inputsMatrix, weightsLayer0, biasesLayer0);

    if (activatedHiddenLayer0)
    {
        *activatedHiddenLayer0 = outputHiddenLayer.vec<NT_HIDDEN_LAYER_SIZE>();
    }
    auto finalOutput =
        ForwardPassGeneric<NT_HIDDEN_LAYER_SIZE, NT_OUTPUT_SIZE, LayerActivation::None>(
            outputHiddenLayer, weightsLayer1, biasesLayer1);
    return finalOutput.vec<NT_OUTPUT_SIZE>();
}

NT_DEVICE inline tcnn::hvec<NT_OUTPUT_SIZE>
ForwardPass12x32x32x16Output(const half *sampledFeatures,
                             const half *weightsLayer0,
                             const half *weightsLayer1,
                             const half *weightsLayer2,
                             const half *biasesLayer0,
                             const half *biasesLayer1,
                             const half *biasesLayer2,
                             tcnn::hvec<NT_MLP_HIDDEN_LAYER0_SIZE> *activatedHiddenLayer0 = 0,
                             tcnn::hvec<NT_MLP_HIDDEN_LAYER1_SIZE> *activatedHiddenLayer1 = 0)
{
    tcnn::hvec<NT_INPUT_SIZE> inputsVector(sampledFeatures);
    tcnn::mma_vec<AlignUp(NT_INPUT_SIZE, 16)> inputsMatrix(inputsVector);

    auto outputHiddenLayer0 =
        ForwardPassGeneric<NT_INPUT_SIZE, NT_MLP_HIDDEN_LAYER0_SIZE>(
            inputsMatrix, weightsLayer0, biasesLayer0);
    if (activatedHiddenLayer0)
    {
        *activatedHiddenLayer0 = outputHiddenLayer0.vec<NT_MLP_HIDDEN_LAYER0_SIZE>();
    }

    auto outputHiddenLayer1 =
        ForwardPassGeneric<NT_MLP_HIDDEN_LAYER0_SIZE, NT_MLP_HIDDEN_LAYER1_SIZE>(
            outputHiddenLayer0, weightsLayer1, biasesLayer1);
    if (activatedHiddenLayer1)
    {
        *activatedHiddenLayer1 = outputHiddenLayer1.vec<NT_MLP_HIDDEN_LAYER1_SIZE>();
    }

    auto finalOutput =
        ForwardPassGeneric<NT_MLP_HIDDEN_LAYER1_SIZE, NT_OUTPUT_SIZE, LayerActivation::None>(
            outputHiddenLayer1, weightsLayer2, biasesLayer2);
    return finalOutput.vec<NT_OUTPUT_SIZE>();
}

template <uint32_t N_THREADS, uint32_t N>
__device__ void SumRows(tcnn::mma_vec<AlignUp(N, 16)> &gradMat,
                        float *sharedPartials, // size: (N_THREADS / 32) * 16
                        float *globalBiasGrad  // size: 16
)
{
    constexpr uint32_t numWarps = N_THREADS / 32;
    constexpr uint32_t alignedSize = AlignUp(N, 16);
    const uint32_t lane = threadIdx.x & 31;
    const uint32_t warp = threadIdx.x >> 5;

    tcnn::hvec<alignedSize> row = gradMat.template vec<alignedSize>();

    float accum[N];
#pragma unroll
    for (int i = 0; i < N; ++i)
    {
        accum[i] = __half2float(row[i]);
    }

#pragma unroll
    for (int i = 0; i < N; ++i)
    {
        accum[i] = WarpReduceSum(accum[i]);
    }

    if (lane == 0)
    {
#pragma unroll
        for (int i = 0; i < N; ++i)
        {
            sharedPartials[warp * N + i] = accum[i];
        }
    }

    __syncthreads();

    if (warp == 0)
    {
#pragma unroll
        for (int i = 0; i < N; ++i)
        {
            float val = lane < numWarps ? sharedPartials[lane * N + i] : 0.f;
            val = WarpReduceSum(val);
            if (lane == 0)
            {
                atomicAdd(&globalBiasGrad[i], val);
            }
        }
    }

    __syncthreads();
}

template <uint32_t N_THREADS, uint32_t M, uint32_t N, tcnn::MatrixLayout LAYOUT>
NT_DEVICE inline void SumIntoLinearGlobalMemoryHierarchicalFloat(
    const tcnn::mma_mat<M, N, LAYOUT> &matrix, float *sharedMemory, float *globalWeightGrad)
{
    static_assert(N_THREADS % 32 == 0, "N_THREADS must be divisible by warp size.");

    using mat_t = tcnn::mma_mat<M, N, LAYOUT>;
    const uint32_t lane = threadIdx.x & 31;
    const uint32_t warpId = threadIdx.x >> 5;
    constexpr uint32_t numWarps = N_THREADS / 32;

    float accum[mat_t::N_REGS * 2];
#pragma unroll
    for (uint32_t i = 0; i < mat_t::N_REGS; ++i)
    {
        const __half2 reg = matrix.regs[i];
        accum[2 * i + 0] = __half2float(__low2half(reg));
        accum[2 * i + 1] = __half2float(__high2half(reg));
    }

    if constexpr (numWarps > 1)
    {
#pragma unroll
        for (uint32_t j = 2; j <= numWarps; j <<= 1)
        {
            const uint32_t sharedBase = (warpId / j) * mat_t::N_ELEMS;

            if (warpId % j == j / 2)
            {
#pragma unroll
                for (uint32_t i = 0; i < mat_t::N_REGS; ++i)
                {
                    const uint32_t linearIndex = mat_t::to_linear(lane, i);
                    sharedMemory[sharedBase + linearIndex + 0] = accum[2 * i + 0];
                    sharedMemory[sharedBase + linearIndex + 1] = accum[2 * i + 1];
                }
            }

            __syncthreads();

            if (warpId % j == 0)
            {
#pragma unroll
                for (uint32_t i = 0; i < mat_t::N_REGS; ++i)
                {
                    const uint32_t linearIndex = mat_t::to_linear(lane, i);
                    accum[2 * i + 0] += sharedMemory[sharedBase + linearIndex + 0];
                    accum[2 * i + 1] += sharedMemory[sharedBase + linearIndex + 1];
                }
            }

            __syncthreads();
        }
    }

    if (warpId == 0)
    {
#pragma unroll
        for (uint32_t i = 0; i < mat_t::N_REGS; ++i)
        {
            const uint32_t linearIndex = mat_t::to_linear(lane, i);
            atomicAdd(&globalWeightGrad[linearIndex + 0], accum[2 * i + 0]);
            atomicAdd(&globalWeightGrad[linearIndex + 1], accum[2 * i + 1]);
        }
    }
}

template <uint32_t numThreads,
          uint32_t inputSize,
          uint32_t outputSize,
          LayerActivation activation = LayerActivation::ReLU>
NT_DEVICE inline tcnn::mma_vec<AlignUp(inputSize, 16)>
BackwardPassGeneric(tcnn::mma_vec<AlignUp(outputSize, 16)> &outputGradientMatrix,
                    const tcnn::hvec<inputSize> &layerInput,
                    const half *weights,
                    float *layerWeightGradients,
                    float *layerBiasGradients,
                    float *shmem)
{
    constexpr int weightsRows = AlignUp(inputSize, 16);
    constexpr int weightsColumns = AlignUp(outputSize, 16);
    using weightsMatrix = tcnn::mma_mat<weightsRows, weightsColumns, tcnn::CM>;

    SumRows<numThreads, outputSize>(outputGradientMatrix, shmem, layerBiasGradients);

    weightsMatrix weightsOutput = weightsMatrix::from_linear_memory(weights);
    auto inputGradientMatrix = outputGradientMatrix * weightsOutput.transpose();

    tcnn::mma_vec<weightsRows> layerMatrixInput(layerInput);
    if constexpr (activation == LayerActivation::ReLU)
    {
        inputGradientMatrix.activate_bwd<tcnn::Activation::ReLU>(layerMatrixInput);
    }

    auto outputWeightGradientMatrix =
        tcnn::outer_product(layerMatrixInput, outputGradientMatrix).flip_layout();

    // Write to memory
    SumIntoLinearGlobalMemoryHierarchicalFloat<numThreads>(
        outputWeightGradientMatrix, shmem, layerWeightGradients);

    return inputGradientMatrix;
}

template <uint32_t numThreads>
NT_DEVICE inline tcnn::hvec<NT_INPUT_SIZE>
BackwardPass12x16xOutput(const tcnn::hvec<NT_OUTPUT_SIZE> &lossGradientVector,
                         const tcnn::hvec<NT_HIDDEN_LAYER_SIZE> &activatedHiddenLayer0,
                         const tcnn::hvec<NT_INPUT_SIZE> &networkInput,
                         const half *weightsHidden0,
                         const half *weightsHidden1,
                         float *layer0WeightGradients,
                         float *layer1WeightGradients,
                         float *layer0BiasGradients,
                         float *layer1BiasGradients)
{
    constexpr uint32_t numWarps = numThreads / 32;
    constexpr uint32_t maxBiasSharedFloats = numWarps * NT_HIDDEN_LAYER_SIZE;
    constexpr uint32_t maxWeightSharedFloats =
        (numWarps > 1 ? (numWarps / 2) * tcnn::mma_mat<16, 16, tcnn::CM>::N_ELEMS : 0);
    constexpr uint32_t shmemFloats =
        maxBiasSharedFloats > maxWeightSharedFloats ? maxBiasSharedFloats : maxWeightSharedFloats;
    __shared__ float shmem[shmemFloats > 0 ? shmemFloats : 1];

    tcnn::mma_vec<NT_OUTPUT_SIZE> outputGradientMatrix(lossGradientVector);
    tcnn::mma_vec<NT_HIDDEN_LAYER_SIZE> hiddenGradientMatrix =
        BackwardPassGeneric<numThreads, NT_HIDDEN_LAYER_SIZE, NT_OUTPUT_SIZE>(
            outputGradientMatrix,
            activatedHiddenLayer0,
            weightsHidden1,
            layer1WeightGradients,
            layer1BiasGradients,
            shmem);

    tcnn::mma_vec<NT_HIDDEN_LAYER_SIZE> inputGradientMatrix =
        BackwardPassGeneric<numThreads,
                            NT_INPUT_SIZE,
                            NT_HIDDEN_LAYER_SIZE,
                            LayerActivation::None>(hiddenGradientMatrix,
                                                   networkInput,
                                                   weightsHidden0,
                                                   layer0WeightGradients,
                                                   layer0BiasGradients,
                                                   shmem);
    return inputGradientMatrix.vec<NT_INPUT_SIZE>();
}

template <uint32_t numThreads>
NT_DEVICE inline tcnn::hvec<NT_INPUT_SIZE>
BackwardPass12x32x32x16Output(const tcnn::hvec<NT_OUTPUT_SIZE> &lossGradientVector,
                              const tcnn::hvec<NT_MLP_HIDDEN_LAYER0_SIZE> &activatedHiddenLayer0,
                              const tcnn::hvec<NT_MLP_HIDDEN_LAYER1_SIZE> &activatedHiddenLayer1,
                              const tcnn::hvec<NT_INPUT_SIZE> &networkInput,
                              const half *weightsHidden0,
                              const half *weightsHidden1,
                              const half *weightsHidden2,
                              float *layer0WeightGradients,
                              float *layer1WeightGradients,
                              float *layer2WeightGradients,
                              float *layer0BiasGradients,
                              float *layer1BiasGradients,
                              float *layer2BiasGradients)
{
    constexpr uint32_t numWarps = numThreads / 32;
    constexpr uint32_t maxBiasSharedFloats = numWarps * NT_MLP_HIDDEN_LAYER0_SIZE;
    constexpr uint32_t maxWeightSharedFloats =
        (numWarps > 1
             ? (numWarps / 2) *
                   tcnn::mma_mat<AlignUp(NT_MLP_HIDDEN_LAYER0_SIZE, 16),
                                 AlignUp(NT_MLP_HIDDEN_LAYER1_SIZE, 16),
                                 tcnn::CM>::N_ELEMS
             : 0);
    constexpr uint32_t shmemFloats =
        maxBiasSharedFloats > maxWeightSharedFloats ? maxBiasSharedFloats : maxWeightSharedFloats;
    __shared__ float shmem[shmemFloats > 0 ? shmemFloats : 1];

    tcnn::mma_vec<NT_OUTPUT_SIZE> outputGradientMatrix(lossGradientVector);
    tcnn::mma_vec<NT_MLP_HIDDEN_LAYER1_SIZE> hiddenGradientMatrix1 =
        BackwardPassGeneric<numThreads, NT_MLP_HIDDEN_LAYER1_SIZE, NT_OUTPUT_SIZE>(
            outputGradientMatrix,
            activatedHiddenLayer1,
            weightsHidden2,
            layer2WeightGradients,
            layer2BiasGradients,
            shmem);

    tcnn::mma_vec<NT_MLP_HIDDEN_LAYER0_SIZE> hiddenGradientMatrix0 =
        BackwardPassGeneric<numThreads,
                            NT_MLP_HIDDEN_LAYER0_SIZE,
                            NT_MLP_HIDDEN_LAYER1_SIZE>(hiddenGradientMatrix1,
                                                       activatedHiddenLayer0,
                                                       weightsHidden1,
                                                       layer1WeightGradients,
                                                       layer1BiasGradients,
                                                       shmem);

    tcnn::mma_vec<AlignUp(NT_INPUT_SIZE, 16)> inputGradientMatrix =
        BackwardPassGeneric<numThreads,
                            NT_INPUT_SIZE,
                            NT_MLP_HIDDEN_LAYER0_SIZE,
                            LayerActivation::None>(hiddenGradientMatrix0,
                                                   networkInput,
                                                   weightsHidden0,
                                                   layer0WeightGradients,
                                                   layer0BiasGradients,
                                                   shmem);
    return inputGradientMatrix.vec<NT_INPUT_SIZE>();
}

} // namespace neural_textures
