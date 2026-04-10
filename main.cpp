#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "train.h"
#if NT_HAS_CUDA
#include <cuda_runtime.h>

#ifndef TEXR_ASSERT
#define TEXR_ASSERT(x) assert(x)
#endif

#define TINYEXR_IMPLEMENTATION
#include <tinyexr.h>
#endif

namespace
{

std::string ToLower(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return (char)std::tolower(c);
    });
    return value;
}

void PrintUsage(const char *programName)
{
    std::printf("Usage: %s <texture0.exr> [texture1.exr ...]\n", programName);
}

#if NT_HAS_CUDA

struct HostTexture
{
    std::filesystem::path path;
    int width = 0;
    int height = 0;
    int numChannels = 0;
    int cudaChannels = 0;
    std::vector<std::string> channelNames;
    std::vector<float> pixels;
};

struct UploadedTexture
{
    std::filesystem::path path;
    int width = 0;
    int height = 0;
    int numChannels = 0;
    int cudaChannels = 0;
    cudaArray_t array = nullptr;
    cudaTextureObject_t texture = 0;
};

void CheckCuda(cudaError_t result, const char *operation)
{
    if (result != cudaSuccess)
    {
        throw std::runtime_error(std::string(operation) +
                                 " failed: " + cudaGetErrorString(result));
    }
}

int GetCudaChannelCount(int numChannels)
{
    if (numChannels == 3)
    {
        return 4;
    }

    return numChannels;
}

HostTexture LoadExrTexture(const std::filesystem::path &path)
{
    EXRVersion exrVersion;
    EXRHeader exrHeader;

    InitEXRHeader(&exrHeader);

    const char *errorMessage = nullptr;

    int result = ParseEXRVersionFromFile(&exrVersion, path.string().c_str());
    if (result != TINYEXR_SUCCESS)
    {
        throw std::runtime_error("Failed to parse EXR version: " + path.string());
    }

    if (exrVersion.multipart || exrVersion.non_image)
    {
        throw std::runtime_error("Multipart or non-image EXRs are not supported: " +
                                 path.string());
    }

    result = ParseEXRHeaderFromFile(&exrHeader, &exrVersion, path.string().c_str(), &errorMessage);
    if (result != TINYEXR_SUCCESS)
    {
        std::string message = "Failed to parse EXR header";
        if (errorMessage)
        {
            message += ": ";
            message += errorMessage;
            FreeEXRErrorMessage(errorMessage);
        }
        throw std::runtime_error(message);
    }

    for (int channelIndex = 0; channelIndex < exrHeader.num_channels; ++channelIndex)
    {
        if (exrHeader.pixel_types[channelIndex] == TINYEXR_PIXELTYPE_HALF)
        {
            exrHeader.requested_pixel_types[channelIndex] = TINYEXR_PIXELTYPE_FLOAT;
        }
    }

    float *rgbaData = nullptr;
    int width = 0;
    int height = 0;
    result = LoadEXR(&rgbaData, &width, &height, path.string().c_str(), &errorMessage);
    if (result != TINYEXR_SUCCESS)
    {
        std::string message = "Failed to load EXR image";
        if (errorMessage)
        {
            message += ": ";
            message += errorMessage;
            FreeEXRErrorMessage(errorMessage);
        }

        FreeEXRHeader(&exrHeader);
        throw std::runtime_error(message);
    }

    if (exrHeader.num_channels != 1 && exrHeader.num_channels != 3 && exrHeader.num_channels != 4)
    {
        std::free(rgbaData);
        FreeEXRHeader(&exrHeader);
        throw std::runtime_error("Unsupported EXR channel count " +
                                 std::to_string(exrHeader.num_channels) + ": " + path.string());
    }

    HostTexture texture;
    texture.path = path;
    texture.width = width;
    texture.height = height;
    texture.numChannels = exrHeader.num_channels;
    texture.cudaChannels = GetCudaChannelCount(texture.numChannels);
    texture.pixels.resize((size_t)texture.width * (size_t)texture.height *
                          (size_t)texture.cudaChannels);
    texture.channelNames.reserve((size_t)texture.numChannels);

    for (int channelIndex = 0; channelIndex < texture.numChannels; ++channelIndex)
    {
        texture.channelNames.emplace_back(exrHeader.channels[channelIndex].name);
    }

    const size_t pixelCount = (size_t)texture.width * (size_t)texture.height;
    for (size_t pixelIndex = 0; pixelIndex < pixelCount; ++pixelIndex)
    {
        if (texture.numChannels == 1)
        {
            texture.pixels[pixelIndex] = rgbaData[4 * pixelIndex + 0];
        }
        else
        {
            for (int channelIndex = 0; channelIndex < texture.numChannels; ++channelIndex)
            {
                texture.pixels[pixelIndex * (size_t)texture.cudaChannels + (size_t)channelIndex] =
                    rgbaData[4 * pixelIndex + (size_t)channelIndex];
            }

            if (texture.numChannels == 3)
            {
                texture.pixels[pixelIndex * (size_t)texture.cudaChannels + 3] = 1.f;
            }
        }
    }

    std::free(rgbaData);
    FreeEXRHeader(&exrHeader);
    return texture;
}

cudaChannelFormatDesc CreateChannelDesc(int numChannels)
{
    cudaChannelFormatDesc desc = {};
    desc.f = cudaChannelFormatKindFloat;
    desc.x = 32;
    desc.y = numChannels >= 2 ? 32 : 0;
    desc.z = numChannels >= 3 ? 32 : 0;
    desc.w = numChannels >= 4 ? 32 : 0;
    return desc;
}

UploadedTexture UploadTexture(const HostTexture &hostTexture)
{
    UploadedTexture uploadedTexture;
    uploadedTexture.path = hostTexture.path;
    uploadedTexture.width = hostTexture.width;
    uploadedTexture.height = hostTexture.height;
    uploadedTexture.numChannels = hostTexture.numChannels;
    uploadedTexture.cudaChannels = hostTexture.cudaChannels;

    cudaChannelFormatDesc channelDesc = CreateChannelDesc(hostTexture.cudaChannels);
    CheckCuda(cudaMallocArray(&uploadedTexture.array,
                              &channelDesc,
                              (size_t)hostTexture.width,
                              (size_t)hostTexture.height),
              "cudaMallocArray");

    const size_t rowPitch =
        (size_t)hostTexture.width * (size_t)hostTexture.cudaChannels * sizeof(float);
    CheckCuda(cudaMemcpy2DToArray(uploadedTexture.array,
                                  0,
                                  0,
                                  hostTexture.pixels.data(),
                                  rowPitch,
                                  rowPitch,
                                  (size_t)hostTexture.height,
                                  cudaMemcpyHostToDevice),
              "cudaMemcpy2DToArray");

    cudaResourceDesc resourceDesc = {};
    resourceDesc.resType = cudaResourceTypeArray;
    resourceDesc.res.array.array = uploadedTexture.array;

    cudaTextureDesc textureDesc = {};
    textureDesc.addressMode[0] = cudaAddressModeWrap;
    textureDesc.addressMode[1] = cudaAddressModeWrap;
    textureDesc.filterMode = cudaFilterModeLinear;
    textureDesc.readMode = cudaReadModeElementType;
    textureDesc.normalizedCoords = 1;

    try
    {
        CheckCuda(cudaCreateTextureObject(
                      &uploadedTexture.texture, &resourceDesc, &textureDesc, nullptr),
                  "cudaCreateTextureObject");
    }
    catch (...)
    {
        cudaFreeArray(uploadedTexture.array);
        uploadedTexture.array = nullptr;
        throw;
    }

    return uploadedTexture;
}

void DestroyUploadedTextures(std::vector<UploadedTexture> &textures)
{
    for (UploadedTexture &texture : textures)
    {
        if (texture.texture != 0)
        {
            cudaDestroyTextureObject(texture.texture);
            texture.texture = 0;
        }

        if (texture.array != nullptr)
        {
            cudaFreeArray(texture.array);
            texture.array = nullptr;
        }
    }
}

#endif

} // namespace

using namespace neural_textures;

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        PrintUsage(argv[0]);
        return 1;
    }

#if !NT_HAS_CUDA
    std::printf("This build does not have CUDA enabled, so EXR textures cannot be uploaded.\n");
    return 1;
#else
    std::vector<HostTexture> hostTextures;
    hostTextures.reserve((size_t)argc - 1);

    try
    {
        for (int argIndex = 1; argIndex < argc; ++argIndex)
        {
            const std::filesystem::path path = argv[argIndex];
            if (!std::filesystem::exists(path))
            {
                throw std::runtime_error("Input file does not exist: " + path.string());
            }

            if (ToLower(path.extension().string()) != ".exr")
            {
                throw std::runtime_error("Only .exr inputs are supported right now: " +
                                         path.string());
            }

            hostTextures.push_back(LoadExrTexture(path));
        }

        std::vector<UploadedTexture> uploadedTextures;
        uploadedTextures.reserve(hostTextures.size());

        for (const HostTexture &hostTexture : hostTextures)
        {
            uploadedTextures.push_back(UploadTexture(hostTexture));
        }

        for (const UploadedTexture &texture : uploadedTextures)
        {
            std::printf("Loaded %s (%dx%d, %d channel%s, cuda=%d) -> cudaTextureObject_t=%llu\n",
                        texture.path.string().c_str(),
                        texture.width,
                        texture.height,
                        texture.numChannels,
                        texture.numChannels == 1 ? "" : "s",
                        texture.cudaChannels,
                        (unsigned long long)texture.texture);
        }

        DestroyUploadedTextures(uploadedTextures);
        return 0;
    }

    catch (const std::exception &error)
    {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }

    const int unconstrainedThreshold = 5000;
    const int blockFeaturesThreshold = unconstrainedThreshold + 200000;
    const int maxIters = blockFeaturesThreshold + 1000;

    KernelParams params;

    for (int iter = 0; iter < maxIters; iter++)
    {
        TrainingKernelType type;
        if (iter < unconstrainedThreshold)
        {
            type = TrainingKernelType::UNCONSTRAINED;
        }
        else if (iter < blockFeaturesThreshold)
        {
            type = TrainingKernelType::BLOCK_FEATURES;
        }
        else
        {
            type = TrainingKernelType::FINALIZE;
        }

        InvokeTraining(params, type);
        InvokeOptimizeNetwork(params);
        InvokeOptimizeFeatures(params);
        params.step++;
    }
#endif
}
