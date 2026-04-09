#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

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
    std::transform(value.begin(),
                   value.end(),
                   value.begin(),
                   [](unsigned char c) { return (char)std::tolower(c); });
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
    std::vector<float4> pixels;
};

struct UploadedTexture
{
    std::filesystem::path path;
    int width = 0;
    int height = 0;
    cudaArray_t array = nullptr;
    cudaTextureObject_t texture = 0;
};

void CheckCuda(cudaError_t result, const char *operation)
{
    if (result != cudaSuccess)
    {
        throw std::runtime_error(std::string(operation) + " failed: " + cudaGetErrorString(result));
    }
}

HostTexture LoadExrTexture(const std::filesystem::path &path)
{
    float *rgbaData = nullptr;
    int width = 0;
    int height = 0;
    const char *errorMessage = nullptr;

    const int loadResult = LoadEXR(&rgbaData, &width, &height, path.string().c_str(), &errorMessage);
    if (loadResult != TINYEXR_SUCCESS)
    {
        std::string message = "Failed to load EXR";
        if (errorMessage)
        {
            message += ": ";
            message += errorMessage;
            FreeEXRErrorMessage(errorMessage);
        }
        throw std::runtime_error(message);
    }

    HostTexture texture;
    texture.path = path;
    texture.width = width;
    texture.height = height;
    texture.pixels.resize((size_t)width * (size_t)height);

    for (size_t pixelIndex = 0; pixelIndex < texture.pixels.size(); ++pixelIndex)
    {
        texture.pixels[pixelIndex] = make_float4(rgbaData[4 * pixelIndex + 0],
                                                 rgbaData[4 * pixelIndex + 1],
                                                 rgbaData[4 * pixelIndex + 2],
                                                 rgbaData[4 * pixelIndex + 3]);
    }

    std::free(rgbaData);
    return texture;
}

UploadedTexture UploadTexture(const HostTexture &hostTexture)
{
    UploadedTexture uploadedTexture;
    uploadedTexture.path = hostTexture.path;
    uploadedTexture.width = hostTexture.width;
    uploadedTexture.height = hostTexture.height;

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
    CheckCuda(cudaMallocArray(&uploadedTexture.array,
                              &channelDesc,
                              (size_t)hostTexture.width,
                              (size_t)hostTexture.height),
              "cudaMallocArray");

    const size_t rowPitch = (size_t)hostTexture.width * sizeof(float4);
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
                throw std::runtime_error("Only .exr inputs are supported right now: " + path.string());
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
            std::printf("Loaded %s (%dx%d) -> cudaTextureObject_t=%llu\n",
                        texture.path.string().c_str(),
                        texture.width,
                        texture.height,
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
#endif
}
