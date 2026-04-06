#pragma once

#include "util/cuda_util.h"
#include <cuda_fp16.h>
#include <tiny-cuda-nn/mma.h>

#if defined(__CUDACC__)
#define NT_DEVICE __device__
#endif

#define NT_INPUT_SIZE 12
#define NT_HIDDEN_LAYER_SIZE 16
#define NT_OUTPUT_SIZE 8

#define half __half

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
__device__ void SumRows(const tcnn::mma_vec<16> &grad_mat,
                        float *shared_partials, // size: (N_THREADS / 32) * 16
                        float *global_bias_grad // size: 16
)
{
    constexpr uint32_t N_WARPS = N_THREADS / 32;
    const uint32_t lane = threadIdx.x & 31;
    const uint32_t warp = threadIdx.x >> 5;

    tcnn::hvec<16> row = grad_mat.vec<16>();

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
            shared_partials[warp * N + i] = accum[i];
        }
    }

    __syncthreads();

    if (warp == 0)
    {
#pragma unroll
        for (int i = 0; i < N; ++i)
        {
            float val = lane < N_WARPS ? shared_partials[lane * N + i] : 0.f;
            val = WarpReduceSum(val);
            if (lane == 0)
            {
                atomicAdd(&global_bias_grad[i], val);
            }
        }
    }

    __syncthreads();
}

template <uint32_t numThreads>
NT_DEVICE inline auto BackwardPass(const tcnn::hvec<NT_OUTPUT_SIZE> &lossGradientVector,
                                   const tcnn::hvec<NT_HIDDEN_LAYER_SIZE> &activatedHiddenLayer0,
                                   const tcnn::hvec<NT_INPUT_SIZE> &networkInput,
                                   const half *weightsHidden0,
                                   const half *weightsHidden1,
                                   float *layer0WeightGradients,
                                   float *layer1WeightGradients,
                                   float *layer0BiasGradients,
                                   float *layer1BiasGradients)
{
    extern __shared__ float shmem[];
    tcnn::mma_vec<16> lossGradientMatrix(lossGradientVector); // 32x8
    // TODO IMPORTANT: NT_OUTPUT_SIZE can change based on the number of outputs
    SumRows<numThreads, NT_OUTPUT_SIZE>(lossGradientMatrix, shmem, layer1BiasGradients);

    tcnn::mma_mat<16, 16, tcnn::CM> weightsOutput =
        tcnn::mma_mat<16, 16, tcnn::CM>::from_linear_memory(weightsHidden1);     // 16x8
    auto hiddenGradientMatrix = lossGradientMatrix * weightsOutput.transpose();  // 32x16
    tcnn::mma_vec<NT_HIDDEN_LAYER_SIZE> outputLayerInput(activatedHiddenLayer0); // 32x16
    hiddenGradientMatrix.activate_bwd<tcnn::Activation::ReLU>(outputLayerInput);
    SumRows<numThreads, NT_HIDDEN_LAYER_SIZE>(hiddenGradientMatrix, shmem, layer0BiasGradients);

    auto outputWeightGradientMatrix =
        tcnn::outer_product(outputLayerInput, lossGradientMatrix); // 16x8

    // Write to memory
    outputWeightGradientMatrix.sum_into_linear_global_memory_hierarchical<numThreads>(
        layer1WeightGradients);

    tcnn::mma_mat<16, 16, tcnn::CM> weightsHiddenLayer0 =
        tcnn::mma_mat<16, 16, tcnn::CM>::from_linear_memory(weightsHidden0);           // 12x16
    auto inputGradientMatrix = hiddenGradientMatrix * weightsHiddenLayer0.transpose(); // 32x12

    tcnn::mma_vec<16> networkInputMatrix(networkInput); // 32x12
    auto hiddenWeightGradientMatrix =
        tcnn::outer_product(networkInputMatrix, hiddenGradientMatrix); // 12x16
    hiddenWeightGradientMatrix.sum_into_linear_global_memory_hierarchical<numThreads>(
        layer0WeightGradients);

    return inputGradientMatrix;
}

struct AdamConstants
{
    float beta1;
    float beta2;
    float learningRate;
    float decay;
};

template <typename T>
inline T Lerp(T x, T y, T s)
{
    return (T(1) - s) * x + s * y;
}

NT_DEVICE inline void AdamOptimize(AdamConstants &constants,
                                   float &moment1,
                                   float &moment2,
                                   float gradient,
                                   float learningRate,
                                   float invBiasCorrection,
                                   float epsilon,
                                   float weight)
{
    float firstMoment = Lerp(gradient, moment1, constants.beta1);
    float secondMoment = Lerp(gradient * gradient, moment2, constants.beta2);

    // TODO: potential variant is to not correct first moment?
    float mhat = firstMoment * invBiasCorrection;
    float vhat = secondMoment * invBiasCorrection;
    float newWeight = weight - learningRate * mhat / (sqrtf(vhat) + epsilon);
}

} // namespace neural_textures
