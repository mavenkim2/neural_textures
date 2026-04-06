#include "forward_pass_test.h"

#include "../mlp.h"

#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <vector>

constexpr uint32_t WARP_SIZE = tcnn::WARP_SIZE;

namespace neural_textures
{

__global__ static void ForwardPassTestKernel(half *outputs,
                                             const half *inputs,
                                             const half *weightsHidden0,
                                             const half *weightsHidden1,
                                             const half *biasesHidden0,
                                             const half *biasesHidden1)
{
    const uint32_t lane = threadIdx.x & (WARP_SIZE - 1u);
    const half *laneInputs = inputs + lane * NT_INPUT_SIZE;

    tcnn::hvec<NT_OUTPUT_SIZE> outputVec =
        ForwardPass(laneInputs, weightsHidden0, weightsHidden1, biasesHidden0, biasesHidden1);

    for (int i = 0; i < NT_OUTPUT_SIZE; i++)
    {
        outputs[lane * NT_OUTPUT_SIZE + i] = outputVec[i];
    }
}

static bool CheckCuda(cudaError_t result, const char *message)
{
    if (result == cudaSuccess)
    {
        return true;
    }

    std::printf("%s: %s\n", message, cudaGetErrorString(result));
    return false;
}

static void InitializeForwardPassTestData(std::vector<half> &inputs,
                                          std::vector<half> &weightsHidden0,
                                          std::vector<half> &weightsHidden1,
                                          std::vector<half> &biasesHidden0,
                                          std::vector<half> &biasesHidden1)
{
    inputs.resize(WARP_SIZE * NT_INPUT_SIZE);
    weightsHidden0.resize(NT_HIDDEN_LAYER_SIZE * NT_HIDDEN_LAYER_SIZE);
    weightsHidden1.resize(NT_HIDDEN_LAYER_SIZE * NT_HIDDEN_LAYER_SIZE);
    biasesHidden0.resize(NT_HIDDEN_LAYER_SIZE);
    biasesHidden1.resize(NT_HIDDEN_LAYER_SIZE);

    for (int lane = 0; lane < WARP_SIZE; lane++)
    {
        for (int k = 0; k < NT_INPUT_SIZE; k++)
        {
            float value = 0.125f + 0.01171875f * float(lane + 1) + 0.0068359375f * float(k + 1);
            inputs[lane * NT_INPUT_SIZE + k] = __float2half(value);
        }
    }

    for (int outChannel = 0; outChannel < NT_HIDDEN_LAYER_SIZE; outChannel++)
    {
        for (int inChannel = 0; inChannel < NT_HIDDEN_LAYER_SIZE; inChannel++)
        {
            float value = 0.f;
            if (inChannel < NT_INPUT_SIZE)
            {
                value = 0.0625f + 0.0048828125f * float((3 * outChannel + 1) % 7) +
                        0.0029296875f * float((5 * inChannel + 2) % 11);
            }

            weightsHidden0[inChannel + outChannel * NT_HIDDEN_LAYER_SIZE] = __float2half(value);
        }
    }

    for (int outChannel = 0; outChannel < NT_HIDDEN_LAYER_SIZE; outChannel++)
    {
        for (int inChannel = 0; inChannel < NT_HIDDEN_LAYER_SIZE; inChannel++)
        {
            float value = 0.f;
            if (outChannel < NT_OUTPUT_SIZE)
            {
                value = 0.09375f + 0.00341796875f * float((7 * outChannel + 3) % 13) +
                        0.001953125f * float((2 * inChannel + 5) % 9);
            }

            weightsHidden1[inChannel + outChannel * NT_HIDDEN_LAYER_SIZE] = __float2half(value);
        }
    }

    for (int outChannel = 0; outChannel < NT_HIDDEN_LAYER_SIZE; outChannel++)
    {
        float hiddenBias = 0.046875f + 0.00390625f * float((4 * outChannel + 1) % 9);
        biasesHidden0[outChannel] = __float2half(hiddenBias);

        float outputBias = 0.f;
        if (outChannel < NT_OUTPUT_SIZE)
        {
            outputBias = 0.078125f + 0.005859375f * float((5 * outChannel + 2) % 7);
        }
        biasesHidden1[outChannel] = __float2half(outputBias);
    }
}

static void ComputeForwardPassReference(const std::vector<half> &inputs,
                                        const std::vector<half> &weightsHidden0,
                                        const std::vector<half> &weightsHidden1,
                                        const std::vector<half> &biasesHidden0,
                                        const std::vector<half> &biasesHidden1,
                                        std::vector<float> &referenceOutputs)
{
    referenceOutputs.assign(WARP_SIZE * NT_OUTPUT_SIZE, 0.f);

    for (int lane = 0; lane < WARP_SIZE; lane++)
    {
        float hidden[NT_HIDDEN_LAYER_SIZE] = {};

        for (int outChannel = 0; outChannel < NT_HIDDEN_LAYER_SIZE; outChannel++)
        {
            float sum = __half2float(biasesHidden0[outChannel]);
            for (int inChannel = 0; inChannel < NT_HIDDEN_LAYER_SIZE; inChannel++)
            {
                float inputValue = inChannel < NT_INPUT_SIZE
                                       ? __half2float(inputs[lane * NT_INPUT_SIZE + inChannel])
                                       : 0.f;
                float weightValue =
                    __half2float(weightsHidden0[inChannel + outChannel * NT_HIDDEN_LAYER_SIZE]);
                sum += inputValue * weightValue;
            }
            hidden[outChannel] = std::max(sum, 0.f);
        }

        for (int outChannel = 0; outChannel < NT_OUTPUT_SIZE; outChannel++)
        {
            float sum = __half2float(biasesHidden1[outChannel]);
            for (int inChannel = 0; inChannel < NT_HIDDEN_LAYER_SIZE; inChannel++)
            {
                float weightValue =
                    __half2float(weightsHidden1[inChannel + outChannel * NT_HIDDEN_LAYER_SIZE]);
                sum += hidden[inChannel] * weightValue;
            }
            referenceOutputs[lane * NT_OUTPUT_SIZE + outChannel] = sum;
        }
    }
}

bool RunForwardPassTest()
{
    std::vector<half> inputsHost;
    std::vector<half> weightsHidden0Host;
    std::vector<half> weightsHidden1Host;
    std::vector<half> biasesHidden0Host;
    std::vector<half> biasesHidden1Host;
    std::vector<float> referenceOutputs;

    InitializeForwardPassTestData(
        inputsHost, weightsHidden0Host, weightsHidden1Host, biasesHidden0Host, biasesHidden1Host);
    ComputeForwardPassReference(inputsHost,
                                weightsHidden0Host,
                                weightsHidden1Host,
                                biasesHidden0Host,
                                biasesHidden1Host,
                                referenceOutputs);

    half *inputsDevice = nullptr;
    half *weightsHidden0Device = nullptr;
    half *weightsHidden1Device = nullptr;
    half *biasesHidden0Device = nullptr;
    half *biasesHidden1Device = nullptr;
    half *outputsDevice = nullptr;

    const size_t inputsBytes = inputsHost.size() * sizeof(half);
    const size_t weightsHidden0Bytes = weightsHidden0Host.size() * sizeof(half);
    const size_t weightsHidden1Bytes = weightsHidden1Host.size() * sizeof(half);
    const size_t biasesHidden0Bytes = biasesHidden0Host.size() * sizeof(half);
    const size_t biasesHidden1Bytes = biasesHidden1Host.size() * sizeof(half);
    const size_t outputsBytes = WARP_SIZE * NT_OUTPUT_SIZE * sizeof(half);

    bool success =
        CheckCuda(cudaMalloc(&inputsDevice, inputsBytes), "cudaMalloc(inputsDevice)") &&
        CheckCuda(cudaMalloc(&weightsHidden0Device, weightsHidden0Bytes),
                  "cudaMalloc(weightsHidden0Device)") &&
        CheckCuda(cudaMalloc(&weightsHidden1Device, weightsHidden1Bytes),
                  "cudaMalloc(weightsHidden1Device)") &&
        CheckCuda(cudaMalloc(&biasesHidden0Device, biasesHidden0Bytes),
                  "cudaMalloc(biasesHidden0Device)") &&
        CheckCuda(cudaMalloc(&biasesHidden1Device, biasesHidden1Bytes),
                  "cudaMalloc(biasesHidden1Device)") &&
        CheckCuda(cudaMalloc(&outputsDevice, outputsBytes), "cudaMalloc(outputsDevice)") &&
        CheckCuda(cudaMemcpy(inputsDevice, inputsHost.data(), inputsBytes, cudaMemcpyHostToDevice),
                  "cudaMemcpy(inputsDevice)") &&
        CheckCuda(cudaMemcpy(weightsHidden0Device,
                             weightsHidden0Host.data(),
                             weightsHidden0Bytes,
                             cudaMemcpyHostToDevice),
                  "cudaMemcpy(weightsHidden0Device)") &&
        CheckCuda(cudaMemcpy(weightsHidden1Device,
                             weightsHidden1Host.data(),
                             weightsHidden1Bytes,
                             cudaMemcpyHostToDevice),
                  "cudaMemcpy(weightsHidden1Device)") &&
        CheckCuda(cudaMemcpy(biasesHidden0Device,
                             biasesHidden0Host.data(),
                             biasesHidden0Bytes,
                             cudaMemcpyHostToDevice),
                  "cudaMemcpy(biasesHidden0Device)") &&
        CheckCuda(cudaMemcpy(biasesHidden1Device,
                             biasesHidden1Host.data(),
                             biasesHidden1Bytes,
                             cudaMemcpyHostToDevice),
                  "cudaMemcpy(biasesHidden1Device)");

    std::vector<half> outputsHost(WARP_SIZE * NT_OUTPUT_SIZE, __float2half(0.f));

    if (success)
    {
        ForwardPassTestKernel<<<1, WARP_SIZE>>>(outputsDevice,
                                                inputsDevice,
                                                weightsHidden0Device,
                                                weightsHidden1Device,
                                                biasesHidden0Device,
                                                biasesHidden1Device);
        success =
            CheckCuda(cudaGetLastError(), "ForwardPassTestKernel launch") &&
            CheckCuda(cudaDeviceSynchronize(), "ForwardPassTestKernel sync") &&
            CheckCuda(cudaMemcpy(
                          outputsHost.data(), outputsDevice, outputsBytes, cudaMemcpyDeviceToHost),
                      "cudaMemcpy(outputsHost)");
    }

    const float tolerance = 2e-2f;
    if (success)
    {
        for (int lane = 0; lane < WARP_SIZE && success; lane++)
        {
            for (int i = 0; i < NT_OUTPUT_SIZE; i++)
            {
                float actual = __half2float(outputsHost[lane * NT_OUTPUT_SIZE + i]);
                float expected = referenceOutputs[lane * NT_OUTPUT_SIZE + i];
                float diff = std::fabs(actual - expected);
                if (diff > tolerance)
                {
                    std::printf("ForwardPassTest mismatch at lane=%d output=%d actual=%f "
                                "expected=%f diff=%f\n",
                                lane,
                                i,
                                actual,
                                expected,
                                diff);
                    success = false;
                    break;
                }
            }
        }
    }

    cudaFree(outputsDevice);
    cudaFree(biasesHidden1Device);
    cudaFree(biasesHidden0Device);
    cudaFree(weightsHidden1Device);
    cudaFree(weightsHidden0Device);
    cudaFree(inputsDevice);

    if (success)
    {
        std::printf("ForwardPassTest passed\n");
    }

    return success;
}

} // namespace neural_textures
