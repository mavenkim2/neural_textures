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
    int numMipLevels = 0;
    std::vector<std::string> channelNames;
    std::vector<std::vector<float>> mipPixels;
};

struct UploadedTexture
{
    std::filesystem::path path;
    int width = 0;
    int height = 0;
    int numChannels = 0;
    int cudaChannels = 0;
    int numMipLevels = 0;
    cudaMipmappedArray_t mipmappedArray = nullptr;
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

int GetMipDimension(int baseDimension, int level, bool roundUp)
{
    if (roundUp)
    {
        const int divisor = 1 << level;
        return std::max(1, (baseDimension + divisor - 1) / divisor);
    }

    return std::max(1, baseDimension >> level);
}

bool ChannelLoadsAsUint(const EXRHeader &header, int channelIndex)
{
    return header.requested_pixel_types[channelIndex] == TINYEXR_PIXELTYPE_UINT;
}

void CopyExrPixelToMip(const EXRHeader &header,
                       unsigned char **images,
                       int numChannels,
                       int cudaChannels,
                       int sourceIndex,
                       float *destPixel)
{
    if (numChannels == 1)
    {
        if (ChannelLoadsAsUint(header, 0))
        {
            destPixel[0] = (float)reinterpret_cast<unsigned int **>(images)[0][sourceIndex];
        }
        else
        {
            destPixel[0] = reinterpret_cast<float **>(images)[0][sourceIndex];
        }
        return;
    }

    for (int channelIndex = 0; channelIndex < numChannels; ++channelIndex)
    {
        if (ChannelLoadsAsUint(header, channelIndex))
        {
            destPixel[channelIndex] =
                (float)reinterpret_cast<unsigned int **>(images)[channelIndex][sourceIndex];
        }
        else
        {
            destPixel[channelIndex] =
                reinterpret_cast<float **>(images)[channelIndex][sourceIndex];
        }
    }

    if (numChannels == 3 && cudaChannels == 4)
    {
        destPixel[3] = 1.f;
    }
}

std::vector<std::vector<float>> LoadExrMipChainFromTiles(const std::filesystem::path &path,
                                                         const EXRHeader &exrHeader)
{
    EXRImage exrImage;
    InitEXRImage(&exrImage);

    const char *errorMessage = nullptr;
    const int result =
        LoadEXRImageFromFile(&exrImage, &exrHeader, path.string().c_str(), &errorMessage);
    if (result != TINYEXR_SUCCESS)
    {
        std::string message = "Failed to load tiled EXR image";
        if (errorMessage)
        {
            message += ": ";
            message += errorMessage;
            FreeEXRErrorMessage(errorMessage);
        }
        throw std::runtime_error(message);
    }

    int maxLevel = 0;
    for (int tileIndex = 0; tileIndex < exrImage.num_tiles; ++tileIndex)
    {
        const EXRTile &tile = exrImage.tiles[tileIndex];
        if (tile.level_x != tile.level_y)
        {
            FreeEXRImage(&exrImage);
            throw std::runtime_error("Ripmap EXRs are not supported: " + path.string());
        }

        maxLevel = std::max(maxLevel, tile.level_x);
    }

    const int cudaChannels = GetCudaChannelCount(exrHeader.num_channels);
    const bool roundUp = exrHeader.tile_rounding_mode == TINYEXR_TILE_ROUND_UP;
    std::vector<std::vector<float>> mipPixels((size_t)maxLevel + 1);

    for (int mipLevel = 0; mipLevel <= maxLevel; ++mipLevel)
    {
        const int mipWidth = GetMipDimension(exrImage.width, mipLevel, roundUp);
        const int mipHeight = GetMipDimension(exrImage.height, mipLevel, roundUp);
        mipPixels[(size_t)mipLevel].resize(
            (size_t)mipWidth * (size_t)mipHeight * (size_t)cudaChannels, 0.f);
    }

    for (int tileIndex = 0; tileIndex < exrImage.num_tiles; ++tileIndex)
    {
        const EXRTile &tile = exrImage.tiles[tileIndex];
        const int mipLevel = tile.level_x;
        const int mipWidth = GetMipDimension(exrImage.width, mipLevel, roundUp);
        float *mipData = mipPixels[(size_t)mipLevel].data();

        for (int tileY = 0; tileY < tile.height; ++tileY)
        {
            for (int tileX = 0; tileX < tile.width; ++tileX)
            {
                const int destX = tile.offset_x * exrHeader.tile_size_x + tileX;
                const int destY = tile.offset_y * exrHeader.tile_size_y + tileY;
                const int sourceIndex = tileY * tile.width + tileX;
                float *destPixel = &mipData[((size_t)destY * (size_t)mipWidth + (size_t)destX) *
                                            (size_t)cudaChannels];
                CopyExrPixelToMip(exrHeader,
                                  tile.images,
                                  exrHeader.num_channels,
                                  cudaChannels,
                                  sourceIndex,
                                  destPixel);
            }
        }
    }

    FreeEXRImage(&exrImage);
    return mipPixels;
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
    texture.channelNames.reserve((size_t)texture.numChannels);

    for (int channelIndex = 0; channelIndex < texture.numChannels; ++channelIndex)
    {
        texture.channelNames.emplace_back(exrHeader.channels[channelIndex].name);
    }

    if (exrHeader.tiled && exrHeader.tile_level_mode == TINYEXR_TILE_MIPMAP_LEVELS)
    {
        texture.mipPixels = LoadExrMipChainFromTiles(path, exrHeader);
        texture.numMipLevels = (int)texture.mipPixels.size();
    }
    else
    {
        texture.numMipLevels = 1;
        texture.mipPixels.resize(1);
        texture.mipPixels[0].resize((size_t)texture.width * (size_t)texture.height *
                                    (size_t)texture.cudaChannels);

        const size_t pixelCount = (size_t)texture.width * (size_t)texture.height;
        for (size_t pixelIndex = 0; pixelIndex < pixelCount; ++pixelIndex)
        {
            if (texture.numChannels == 1)
            {
                texture.mipPixels[0][pixelIndex] = rgbaData[4 * pixelIndex + 0];
            }
            else
            {
                for (int channelIndex = 0; channelIndex < texture.numChannels; ++channelIndex)
                {
                    texture.mipPixels[0][pixelIndex * (size_t)texture.cudaChannels +
                                         (size_t)channelIndex] =
                        rgbaData[4 * pixelIndex + (size_t)channelIndex];
                }

                if (texture.numChannels == 3)
                {
                    texture.mipPixels[0][pixelIndex * (size_t)texture.cudaChannels + 3] = 1.f;
                }
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
    uploadedTexture.numMipLevels = hostTexture.numMipLevels;

    cudaChannelFormatDesc channelDesc = CreateChannelDesc(hostTexture.cudaChannels);
    CheckCuda(cudaMallocMipmappedArray(
                  &uploadedTexture.mipmappedArray,
                  &channelDesc,
                  make_cudaExtent((size_t)hostTexture.width, (size_t)hostTexture.height, 0),
                  (unsigned int)hostTexture.numMipLevels),
              "cudaMallocMipmappedArray");

    for (int mipLevel = 0; mipLevel < hostTexture.numMipLevels; ++mipLevel)
    {
        cudaArray_t levelArray = nullptr;
        CheckCuda(cudaGetMipmappedArrayLevel(
                      &levelArray, uploadedTexture.mipmappedArray, (unsigned int)mipLevel),
                  "cudaGetMipmappedArrayLevel");

        const int mipWidth = GetMipDimension(hostTexture.width, mipLevel, false);
        const int mipHeight = GetMipDimension(hostTexture.height, mipLevel, false);
        const size_t rowPitch =
            (size_t)mipWidth * (size_t)hostTexture.cudaChannels * sizeof(float);
        CheckCuda(cudaMemcpy2DToArray(levelArray,
                                      0,
                                      0,
                                      hostTexture.mipPixels[(size_t)mipLevel].data(),
                                      rowPitch,
                                      rowPitch,
                                      (size_t)mipHeight,
                                      cudaMemcpyHostToDevice),
                  "cudaMemcpy2DToArray");
    }

    cudaResourceDesc resourceDesc = {};
    resourceDesc.resType = cudaResourceTypeMipmappedArray;
    resourceDesc.res.mipmap.mipmap = uploadedTexture.mipmappedArray;

    cudaTextureDesc textureDesc = {};
    textureDesc.addressMode[0] = cudaAddressModeWrap;
    textureDesc.addressMode[1] = cudaAddressModeWrap;
    textureDesc.filterMode = cudaFilterModeLinear;
    textureDesc.mipmapFilterMode = cudaFilterModeLinear;
    textureDesc.readMode = cudaReadModeElementType;
    textureDesc.normalizedCoords = 1;
    textureDesc.minMipmapLevelClamp = 0.0f;
    textureDesc.maxMipmapLevelClamp = (float)(hostTexture.numMipLevels - 1);

    try
    {
        CheckCuda(cudaCreateTextureObject(
                      &uploadedTexture.texture, &resourceDesc, &textureDesc, nullptr),
                  "cudaCreateTextureObject");
    }
    catch (...)
    {
        cudaFreeMipmappedArray(uploadedTexture.mipmappedArray);
        uploadedTexture.mipmappedArray = nullptr;
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

        if (texture.mipmappedArray != nullptr)
        {
            cudaFreeMipmappedArray(texture.mipmappedArray);
            texture.mipmappedArray = nullptr;
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
            std::printf(
                "Loaded %s (%dx%d, %d channel%s, cuda=%d, mips=%d) -> cudaTextureObject_t=%llu\n",
                texture.path.string().c_str(),
                texture.width,
                texture.height,
                texture.numChannels,
                texture.numChannels == 1 ? "" : "s",
                texture.cudaChannels,
                texture.numMipLevels,
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

    // 1. what rng am i using?
    // 2. how do I initialize the features?
    // 3. actually sampling the reference textures during training
    // 4. set feature sizes
#if 1
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
        if (type != TrainingKernelType::FINALIZE)
        {
            InvokeOptimizeFeatures(params);
        }
        params.step++;
    }
#endif
#endif
}
