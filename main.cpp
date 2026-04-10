#include <algorithm>
#include <array>
#include <cassert>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <random>
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
    std::printf("Usage: %s [--mode <0-3>] <texture0.exr> [texture1.exr ...]\n", programName);
    std::printf("  mode 0 = BCf-0.5k\n");
    std::printf("  mode 1 = BCf-1k\n");
    std::printf("  mode 2 = BCf-2k (default)\n");
    std::printf("  mode 3 = BCf-2k++\n");
}

#if NT_HAS_CUDA

using namespace neural_textures;

struct CommandLineOptions
{
    int mode = 2;
    std::vector<std::filesystem::path> inputPaths;
};

struct FeatureModeConfig
{
    int resolution = 0;
    int numMips = 0;
};

struct TrainingModeConfig
{
    const char *name = "";
    std::array<FeatureModeConfig, NT_NUM_FEATURES> features;
};

constexpr TrainingModeConfig gTrainingModeConfigs[] = {
    {"BCf-0.5k", {{{512, 8}, {256, 7}, {128, 6}, {64, 5}}}},
    {"BCf-1k", {{{1024, 9}, {512, 8}, {256, 7}, {128, 6}}}},
    {"BCf-2k", {{{2048, 10}, {1024, 9}, {512, 8}, {256, 7}}}},
    {"BCf-2k++", {{{2048, 10}, {2048, 10}, {512, 8}, {256, 7}}}},
};

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

CommandLineOptions ParseCommandLine(int argc, char *argv[])
{
    CommandLineOptions options;

    for (int argIndex = 1; argIndex < argc; ++argIndex)
    {
        const std::string argument = argv[argIndex];
        if (argument == "--mode" || argument == "-m")
        {
            if (argIndex + 1 >= argc)
            {
                throw std::runtime_error("Expected an integer after --mode");
            }

            options.mode = std::stoi(argv[++argIndex]);
            continue;
        }

        if (argument.rfind("--mode=", 0) == 0)
        {
            options.mode = std::stoi(argument.substr(7));
            continue;
        }

        options.inputPaths.emplace_back(argv[argIndex]);
    }

    if (options.mode < 0 || options.mode >= (int)std::size(gTrainingModeConfigs))
    {
        throw std::runtime_error("Mode must be in the range [0, 3]");
    }

    return options;
}

template <typename T>
T *AllocateDeviceBuffer(size_t count, const char *operation)
{
    T *buffer = nullptr;
    CheckCuda(cudaMalloc((void **)&buffer, count * sizeof(T)), operation);
    return buffer;
}

template <typename T>
void ZeroDeviceBuffer(T *buffer, size_t count, const char *operation)
{
    CheckCuda(cudaMemset(buffer, 0, count * sizeof(T)), operation);
}

template <typename T>
T **UploadPointerArray(const std::vector<T *> &hostPointers, const char *operation)
{
    T **devicePointers = AllocateDeviceBuffer<T *>(hostPointers.size(), operation);
    CheckCuda(cudaMemcpy(devicePointers,
                         hostPointers.data(),
                         hostPointers.size() * sizeof(T *),
                         cudaMemcpyHostToDevice),
              operation);
    return devicePointers;
}

template <typename T>
void UploadHostVector(T *deviceBuffer, const std::vector<T> &hostData, const char *operation)
{
    CheckCuda(
        cudaMemcpy(
            deviceBuffer, hostData.data(), hostData.size() * sizeof(T), cudaMemcpyHostToDevice),
        operation);
}

std::vector<half>
InitializeNetworkWeights(size_t totalCount, size_t activeCount, std::mt19937 &rng, float scale)
{
    std::uniform_real_distribution<float> distribution(-scale, scale);
    std::vector<half> weights(totalCount, __float2half(0.f));
    for (size_t index = 0; index < activeCount; ++index)
    {
        weights[index] = __float2half(distribution(rng));
    }

    return weights;
}

std::vector<float3> InitializeFloat3Texels(size_t texelCount, std::mt19937 &rng)
{
    std::uniform_real_distribution<float> distribution(0.f, 1.f);
    std::vector<float3> texels(texelCount);

    for (float3 &texel : texels)
    {
        texel = make_float3(distribution(rng), distribution(rng), distribution(rng));
    }

    return texels;
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

int GetBlockCountForDimension(int size)
{
    int numBlocks = size >> 2;
    return numBlocks > 0 ? numBlocks : 1;
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

void InitializeFeatureStorage(Feature &feature, const FeatureModeConfig &config, std::mt19937 &rng)
{
    feature.width = config.resolution;
    feature.height = config.resolution;
    feature.numMips = config.numMips;

    std::vector<BC6Parameters *> gridPointers((size_t)feature.numMips);
    std::vector<BC6ParameterGradients *> gradientPointers((size_t)feature.numMips);
    std::vector<BC6ParameterGradients *> moment1Pointers((size_t)feature.numMips);
    std::vector<BC6ParameterGradients *> moment2Pointers((size_t)feature.numMips);
    std::vector<float3 *> unconstrainedGridPointers((size_t)feature.numMips);
    std::vector<float3 *> unconstrainedGradientPointers((size_t)feature.numMips);
    std::vector<float3 *> unconstrainedMoment1Pointers((size_t)feature.numMips);
    std::vector<float3 *> unconstrainedMoment2Pointers((size_t)feature.numMips);

    for (int mip = 0; mip < feature.numMips; ++mip)
    {
        const int mipWidth = std::max(feature.width >> mip, 1);
        const int mipHeight = std::max(feature.height >> mip, 1);
        const size_t texelCount = (size_t)mipWidth * (size_t)mipHeight;
        const size_t blockCount = (size_t)GetBlockCountForDimension(mipWidth) *
                                  (size_t)GetBlockCountForDimension(mipHeight);

        gridPointers[(size_t)mip] =
            AllocateDeviceBuffer<BC6Parameters>(blockCount, "cudaMalloc(feature.grid[mip])");
        gradientPointers[(size_t)mip] = AllocateDeviceBuffer<BC6ParameterGradients>(
            blockCount, "cudaMalloc(feature.gradients[mip])");
        moment1Pointers[(size_t)mip] = AllocateDeviceBuffer<BC6ParameterGradients>(
            blockCount, "cudaMalloc(feature.moment1[mip])");
        moment2Pointers[(size_t)mip] = AllocateDeviceBuffer<BC6ParameterGradients>(
            blockCount, "cudaMalloc(feature.moment2[mip])");

        ZeroDeviceBuffer(gridPointers[(size_t)mip], blockCount, "cudaMemset(feature.grid[mip])");
        ZeroDeviceBuffer(
            gradientPointers[(size_t)mip], blockCount, "cudaMemset(feature.gradients[mip])");
        ZeroDeviceBuffer(
            moment1Pointers[(size_t)mip], blockCount, "cudaMemset(feature.moment1[mip])");
        ZeroDeviceBuffer(
            moment2Pointers[(size_t)mip], blockCount, "cudaMemset(feature.moment2[mip])");

        unconstrainedGridPointers[(size_t)mip] =
            AllocateDeviceBuffer<float3>(texelCount, "cudaMalloc(feature.unconstrainedGrid[mip])");
        unconstrainedGradientPointers[(size_t)mip] = AllocateDeviceBuffer<float3>(
            texelCount, "cudaMalloc(feature.unconstrainedGradients[mip])");
        unconstrainedMoment1Pointers[(size_t)mip] = AllocateDeviceBuffer<float3>(
            texelCount, "cudaMalloc(feature.unconstrainedMoment1[mip])");
        unconstrainedMoment2Pointers[(size_t)mip] = AllocateDeviceBuffer<float3>(
            texelCount, "cudaMalloc(feature.unconstrainedMoment2[mip])");

        std::vector<float3> texels = InitializeFloat3Texels(texelCount, rng);
        UploadHostVector(unconstrainedGridPointers[(size_t)mip],
                         texels,
                         "cudaMemcpy(feature.unconstrainedGrid[mip])");
        ZeroDeviceBuffer(unconstrainedGradientPointers[(size_t)mip],
                         texelCount,
                         "cudaMemset(feature.unconstrainedGradients[mip])");
        ZeroDeviceBuffer(unconstrainedMoment1Pointers[(size_t)mip],
                         texelCount,
                         "cudaMemset(feature.unconstrainedMoment1[mip])");
        ZeroDeviceBuffer(unconstrainedMoment2Pointers[(size_t)mip],
                         texelCount,
                         "cudaMemset(feature.unconstrainedMoment2[mip])");
    }

    feature.grid = UploadPointerArray(gridPointers, "cudaMemcpy(feature.grid)");
    feature.gradients = UploadPointerArray(gradientPointers, "cudaMemcpy(feature.gradients)");
    feature.moment1 = UploadPointerArray(moment1Pointers, "cudaMemcpy(feature.moment1)");
    feature.moment2 = UploadPointerArray(moment2Pointers, "cudaMemcpy(feature.moment2)");
    feature.unconstrainedGrid =
        UploadPointerArray(unconstrainedGridPointers, "cudaMemcpy(feature.unconstrainedGrid)");
    feature.unconstrainedGradients = UploadPointerArray(
        unconstrainedGradientPointers, "cudaMemcpy(feature.unconstrainedGradients)");
    feature.unconstrainedMoment1 = UploadPointerArray(unconstrainedMoment1Pointers,
                                                      "cudaMemcpy(feature.unconstrainedMoment1)");
    feature.unconstrainedMoment2 = UploadPointerArray(unconstrainedMoment2Pointers,
                                                      "cudaMemcpy(feature.unconstrainedMoment2)");
}

void InitializeNetworkStorage(KernelParams &params, std::mt19937 &rng)
{
    constexpr size_t paddedWeightCount = NT_HIDDEN_LAYER_SIZE * NT_HIDDEN_LAYER_SIZE;
    constexpr size_t biasCount = NT_HIDDEN_LAYER_SIZE;
    const size_t activeWeightCounts[NT_NUM_NETWORK_LAYERS] = {
        NT_HIDDEN_LAYER_SIZE * NT_INPUT_SIZE,
        NT_HIDDEN_LAYER_SIZE * NT_OUTPUT_SIZE,
    };

    for (int layer = 0; layer < NT_NUM_NETWORK_LAYERS; ++layer)
    {
        params.networkWeights[layer] =
            AllocateDeviceBuffer<half>(paddedWeightCount, "cudaMalloc(networkWeights)");
        params.networkWeightGradients[layer] =
            AllocateDeviceBuffer<float>(paddedWeightCount, "cudaMalloc(networkWeightGradients)");
        params.networkWeightMoment1[layer] =
            AllocateDeviceBuffer<float>(paddedWeightCount, "cudaMalloc(networkWeightMoment1)");
        params.networkWeightMoment2[layer] =
            AllocateDeviceBuffer<float>(paddedWeightCount, "cudaMalloc(networkWeightMoment2)");

        std::vector<half> weights =
            InitializeNetworkWeights(paddedWeightCount, activeWeightCounts[layer], rng, 1e-2f);
        UploadHostVector(params.networkWeights[layer], weights, "cudaMemcpy(networkWeights)");
        ZeroDeviceBuffer(params.networkWeightGradients[layer],
                         paddedWeightCount,
                         "cudaMemset(networkWeightGradients)");
        ZeroDeviceBuffer(params.networkWeightMoment1[layer],
                         paddedWeightCount,
                         "cudaMemset(networkWeightMoment1)");
        ZeroDeviceBuffer(params.networkWeightMoment2[layer],
                         paddedWeightCount,
                         "cudaMemset(networkWeightMoment2)");

        params.networkBiases[layer] =
            AllocateDeviceBuffer<half>(biasCount, "cudaMalloc(networkBiases)");
        params.networkBiasGradients[layer] =
            AllocateDeviceBuffer<float>(biasCount, "cudaMalloc(networkBiasGradients)");
        params.networkBiasMoment1[layer] =
            AllocateDeviceBuffer<float>(biasCount, "cudaMalloc(networkBiasMoment1)");
        params.networkBiasMoment2[layer] =
            AllocateDeviceBuffer<float>(biasCount, "cudaMalloc(networkBiasMoment2)");

        ZeroDeviceBuffer(params.networkBiases[layer], biasCount, "cudaMemset(networkBiases)");
        ZeroDeviceBuffer(
            params.networkBiasGradients[layer], biasCount, "cudaMemset(networkBiasGradients)");
        ZeroDeviceBuffer(
            params.networkBiasMoment1[layer], biasCount, "cudaMemset(networkBiasMoment1)");
        ZeroDeviceBuffer(
            params.networkBiasMoment2[layer], biasCount, "cudaMemset(networkBiasMoment2)");
    }
}

void InitializeTrainingMode(KernelParams &params, int mode)
{
    const TrainingModeConfig &config = gTrainingModeConfigs[mode];
    std::mt19937 rng(1337u + (uint32_t)mode);

    params.featureAdam.learningRate = 5e-2f;
    params.featureAdam.beta1 = 0.9f;
    params.featureAdam.beta2 = 0.999f;
    params.featureAdam.epsilon = 1e-8f;

    params.networkAdam.learningRate = 1e-3f;
    params.networkAdam.beta1 = 0.9f;
    params.networkAdam.beta2 = 0.999f;
    params.networkAdam.epsilon = 1e-8f;

    params.numMips = 0;
    for (int featureIndex = 0; featureIndex < NT_NUM_FEATURES; ++featureIndex)
    {
        InitializeFeatureStorage(
            params.features[featureIndex], config.features[(size_t)featureIndex], rng);
        params.numMips = std::max(params.numMips, params.features[featureIndex].numMips);
    }

    InitializeNetworkStorage(params, rng);

    params.imageWidth = config.features[0].resolution;
    params.imageHeight = config.features[0].resolution;
    params.numBlocksU = GetBlockCountForDimension(params.imageWidth);
    params.numBlocksV = GetBlockCountForDimension(params.imageHeight);
    params.numSamples = 512 * 512;
}

void InitializeReferenceTextures(neural_textures::KernelParams &params,
                                 const std::vector<UploadedTexture> &uploadedTextures)
{
    if (uploadedTextures.size() > NT_MAX_REFERENCE_TEXTURES)
    {
        throw std::runtime_error("Too many reference textures for KernelParams: " +
                                 std::to_string(uploadedTextures.size()));
    }

    params.numReferenceTextures = (int)uploadedTextures.size();

    for (int textureIndex = 0; textureIndex < params.numReferenceTextures; ++textureIndex)
    {
        const UploadedTexture &uploadedTexture = uploadedTextures[(size_t)textureIndex];
        neural_textures::ReferenceTexture &referenceTexture =
            params.referenceTextures[textureIndex];
        referenceTexture.texture = uploadedTexture.texture;
        referenceTexture.width = uploadedTexture.width;
        referenceTexture.height = uploadedTexture.height;
        referenceTexture.numChannels = uploadedTexture.numChannels;
        referenceTexture.numMipLevels = uploadedTexture.numMipLevels;
    }
}

#endif

} // namespace

using namespace neural_textures;

int main(int argc, char *argv[])
{
#if !NT_HAS_CUDA
    std::printf("This build does not have CUDA enabled, so EXR textures cannot be uploaded.\n");
    return 1;
#else
    KernelParams params = {};

    try
    {
        CommandLineOptions options = ParseCommandLine(argc, argv);
        if (options.inputPaths.empty())
        {
            PrintUsage(argv[0]);
            return 1;
        }

        std::vector<HostTexture> hostTextures;
        hostTextures.reserve(options.inputPaths.size());

        InitializeTrainingMode(params, options.mode);

        for (const std::filesystem::path &path : options.inputPaths)
        {
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

        InitializeReferenceTextures(params, uploadedTextures);
        params.imageWidth =
            uploadedTextures.empty() ? params.imageWidth : uploadedTextures[0].width;
        params.imageHeight =
            uploadedTextures.empty() ? params.imageHeight : uploadedTextures[0].height;
        if (!uploadedTextures.empty())
        {
            if (params.imageWidth < 512 || params.imageHeight < 512)
            {
                throw std::runtime_error(
                    "Smoke-test training expects reference textures of at least 512x512");
            }

            for (const UploadedTexture &texture : uploadedTextures)
            {
                if (texture.width != params.imageWidth || texture.height != params.imageHeight)
                {
                    throw std::runtime_error("All reference textures must have the same "
                                             "dimensions for smoke-test training");
                }
            }
        }

        const TrainingModeConfig &modeConfig = gTrainingModeConfigs[options.mode];
        std::printf("Mode %d (%s)\n", options.mode, modeConfig.name);
        for (int featureIndex = 0; featureIndex < NT_NUM_FEATURES; ++featureIndex)
        {
            const Feature &feature = params.features[featureIndex];
            std::printf("  T%d: res=%d mips=%d\n", featureIndex, feature.width, feature.numMips);
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

#if 1
        const int unconstrainedThreshold = 5000;
        const int blockFeaturesThreshold = unconstrainedThreshold + 200000;
        const int maxIters = blockFeaturesThreshold + 1000;
        const int progressInterval = 1000;
        const auto trainingStart = std::chrono::steady_clock::now();

        std::printf("Starting training for %d iterations\n", maxIters);

        for (int iter = 0; iter < maxIters; iter++)
        {
            // if (iter == unconstrainedThreshold)
            // {
            // }

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

            type = TrainingKernelType::UNCONSTRAINED;

            InvokeTraining(params, type);
            InvokeOptimizeNetwork(params);
            if (type != TrainingKernelType::FINALIZE)
            {
                InvokeOptimizeFeatures(params);
            }
            params.step++;

            if (((iter + 1) % progressInterval) == 0 || (iter + 1) == maxIters)
            {
                CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(training progress)");
                const auto now = std::chrono::steady_clock::now();
                const double elapsedSeconds =
                    std::chrono::duration<double>(now - trainingStart).count();
                std::printf(
                    "Completed iteration %d / %d (%.2fs)\n", iter + 1, maxIters, elapsedSeconds);
            }
        }
#endif

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
