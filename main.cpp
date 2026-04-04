#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

#include "tests/forward_pass_test.h"

static inline float ReLU(float v)
{
    return std::max(v, 0.f);
}

static inline void ReLU(float *v, int num)
{
    for (int i = 0; i < num; i++)
    {
        v[i] = ReLU(v[i]);
    }
}

struct Layer
{
    float *weights;
    float *bias;
    int inputSize;
    int outputSize;

    Layer(int inputSize, int outputSize);
    void ComputeOutput(const float *inputs, float *outputs) const;
    void Backpropagate() const;
};

Layer::Layer(int inputSize, int outputSize) : inputSize(inputSize), outputSize(outputSize)
{
    weights = (float *)malloc(sizeof(float) * inputSize * outputSize);
    bias = (float *)malloc(sizeof(float) * outputSize);
}

void Layer::ComputeOutput(const float *inputs, float *outputs) const
{
    for (int i = 0; i < outputSize; i++)
    {
        float total = 0.f;
        for (int j = 0; j < inputSize; j++)
        {
            total += weights[i * inputSize + j] * inputs[j];
        }
        total += bias[i];
        outputs[i] = total;
    }
}

void Layer::Backpropagate() const
{
    // L2 loss
    // (target - actual)^2
}

int main(int argc, int argv[])
{
    (void)argc;
    (void)argv;

    if (!neural_textures::RunForwardPassTest())
    {
        std::printf("ForwardPassTest failed\n");
        return 1;
    }

    float *image = 0;
    (void)image;

    // Train
    const int hiddenLayerSize = 16;
    const int inputSize = 12;
    const int outputSize = 9;

    float *inputs = (float *)malloc(sizeof(float) * inputSize);
    float *hiddenLayerOutput = (float *)malloc(sizeof(float) * hiddenLayerSize);
    float *outputs = (float *)malloc(sizeof(float) * outputSize);

    Layer hiddenLayer(inputSize, hiddenLayerSize);
    Layer outputLayer(hiddenLayerSize, outputSize);

    hiddenLayer.ComputeOutput(inputs, hiddenLayerOutput);
    ReLU(hiddenLayerOutput, hiddenLayerSize);

    outputLayer.ComputeOutput(hiddenLayerOutput, outputs);

    // dCost / d0utput = 2.f / numOutputs * (output - target)
    // dOutput / d = 
}
