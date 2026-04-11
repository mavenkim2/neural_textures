#pragma once

#include "train.h"
#include "util/common.h"
#include "util/cuda_util.h"
#include <cmath>
#include <tiny-cuda-nn/mma.h>

namespace neural_textures
{

NT_DEVICE inline tcnn::hvec<NT_OUTPUT_SIZE>
ForwardPass(const half *sampledFeatures,
            const half *weightsLayer0,
            const half *weightsLayer1,
            const half *biasesLayer0,
            const half *biasesLayer1,
            tcnn::hvec<NT_HIDDEN_LAYER_SIZE> *activatedHiddenLayer0 = 0)
{
    tcnn::hvec<NT_INPUT_SIZE> inputsVector0(sampledFeatures);
    tcnn::mma_vec<16> inputsMatrix0(inputsVector0);

    tcnn::mma_mat<16, 16, tcnn::CM> weightsHiddenLayer0 =
        tcnn::mma_mat<16, 16, tcnn::CM>::from_linear_memory(weightsLayer0);
    tcnn::hvec<NT_HIDDEN_LAYER_SIZE> vecBiasesLayer0(biasesLayer0);
    tcnn::mma_vec<NT_HIDDEN_LAYER_SIZE> matBiasesLayer0(vecBiasesLayer0);

    auto outputHiddenLayer0 = madd(inputsMatrix0, weightsHiddenLayer0, matBiasesLayer0);
    outputHiddenLayer0.activate<tcnn::Activation::ReLU>();

    if (activatedHiddenLayer0)
    {
        *activatedHiddenLayer0 = outputHiddenLayer0.vec<NT_HIDDEN_LAYER_SIZE>();
    }

    tcnn::mma_mat<16, 16, tcnn::CM> weightsHiddenLayer1 =
        tcnn::mma_mat<16, 16, tcnn::CM>::from_linear_memory(weightsLayer1);
    tcnn::hvec<NT_HIDDEN_LAYER_SIZE> vecBiasesLayer1(biasesLayer1);
    tcnn::mma_vec<NT_HIDDEN_LAYER_SIZE> matBiasesLayer1(vecBiasesLayer1);

    auto finalOutput = madd(outputHiddenLayer0, weightsHiddenLayer1, matBiasesLayer1);
    // finalOutput.activate<tcnn::Activation::ReLU>();

    return finalOutput.vec<NT_OUTPUT_SIZE>();
}

template <uint32_t N_THREADS, uint32_t N>
__device__ void SumRows(tcnn::mma_vec<16> &gradMat,
                        float *sharedPartials, // size: (N_THREADS / 32) * 16
                        float *globalBiasGrad  // size: 16
)
{
    constexpr uint32_t numWarps = N_THREADS / 32;
    const uint32_t lane = threadIdx.x & 31;
    const uint32_t warp = threadIdx.x >> 5;

    tcnn::hvec<16> row = gradMat.vec<16>();

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

template <uint32_t numThreads>
NT_DEVICE inline tcnn::hvec<NT_INPUT_SIZE>
BackwardPass(const tcnn::hvec<NT_OUTPUT_SIZE> &lossGradientVector,
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

    tcnn::mma_vec<16> lossGradientMatrix(lossGradientVector); // 32xNT_OUTPUT_SIZE
    SumRows<numThreads, NT_OUTPUT_SIZE>(lossGradientMatrix, shmem, layer1BiasGradients);

    tcnn::mma_mat<16, 16, tcnn::CM> weightsOutput =
        tcnn::mma_mat<16, 16, tcnn::CM>::from_linear_memory(weightsHidden1);      // 16x16
    auto hiddenGradientMatrix = lossGradientMatrix * weightsOutput.transpose();   // 32x16
    tcnn::mma_vec<NT_HIDDEN_LAYER_SIZE> outputLayerInput(activatedHiddenLayer0); // 32x16
    hiddenGradientMatrix.activate_bwd<tcnn::Activation::ReLU>(outputLayerInput);
    SumRows<numThreads, NT_HIDDEN_LAYER_SIZE>(hiddenGradientMatrix, shmem, layer0BiasGradients);

    auto outputWeightGradientMatrix =
        tcnn::outer_product(outputLayerInput, lossGradientMatrix)
            .flip_layout(); // 16x16, stored to CM weight memory

    // Write to memory
    SumIntoLinearGlobalMemoryHierarchicalFloat<numThreads>(
        outputWeightGradientMatrix, shmem, layer1WeightGradients);

    tcnn::mma_mat<16, 16, tcnn::CM> weightsHiddenLayer0 =
        tcnn::mma_mat<16, 16, tcnn::CM>::from_linear_memory(weightsHidden0);           // 12x16
    auto inputGradientMatrix = hiddenGradientMatrix * weightsHiddenLayer0.transpose(); // 32x12

    tcnn::mma_vec<16> networkInputMatrix(networkInput); // 32x12
    auto hiddenWeightGradientMatrix =
        tcnn::outer_product(networkInputMatrix, hiddenGradientMatrix)
            .flip_layout(); // 12x16, stored to CM weight memory
    SumIntoLinearGlobalMemoryHierarchicalFloat<numThreads>(
        hiddenWeightGradientMatrix, shmem, layer0WeightGradients);

    return inputGradientMatrix.vec<NT_INPUT_SIZE>();
}

} // namespace neural_textures
