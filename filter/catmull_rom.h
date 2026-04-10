#pragma once

#include "util/common.h"
#include "util/float2.h"
#include "util/float4.h"
#include <cuda_runtime.h>

namespace neural_textures
{

template <typename T>
NT_DEVICE inline T ZeroValue();

template <>
NT_DEVICE inline float ZeroValue<float>()
{
    return 0.0f;
}

template <>
NT_DEVICE inline float2 ZeroValue<float2>()
{
    return make_float2(0.0f, 0.0f);
}

template <>
NT_DEVICE inline float4 ZeroValue<float4>()
{
    return make_float4(0.0f, 0.0f, 0.0f, 0.0f);
}

// The following code is licensed under the MIT license:
// https://gist.github.com/TheRealMJP/bc503b0b87b643d3505d41eab8b332ae

// Samples a texture with Catmull-Rom filtering, using 9 texture fetches instead of 16.
// See http://vec3.ca/bicubic-filtering-in-fewer-taps/ for more details.
template <typename T>
NT_DEVICE inline T
SampleTextureCatmullRomLod(
    cudaTextureObject_t texture, const float2 &uv, const float2 &texSize, float lod)
{
    // We're going to sample a 4x4 grid of texels surrounding the target UV coordinate. We'll do
    // this by rounding down the sample location to get the exact center of our "starting" texel.
    // The starting texel will be at location [1, 1] in the grid, where [0, 0] is the top left
    // corner.
    float2 samplePos = uv * texSize;
    float2 texPos1 = Floor(samplePos - 0.5f) + 0.5f;

    // Compute the fractional offset from our starting texel to our original sample location, which
    // we'll feed into the Catmull-Rom spline function to get our filter weights.
    float2 f = samplePos - texPos1;

    // Compute the Catmull-Rom weights using the fractional offset that we calculated earlier.
    // These equations are pre-expanded based on our knowledge of where the texels will be located,
    // which lets us avoid having to evaluate a piece-wise function.
    float2 w0 = f * (f * (f * -0.5f + 1.0f) - 0.5f);
    float2 w1 = f * f * (f * 1.5f - 2.5f) + 1.0f;
    float2 w2 = f * (f * (f * -1.5f + 2.0f) + 0.5f);
    float2 w3 = f * f * (f * 0.5f - 0.5f);

    // Work out weighting factors and sampling offsets that will let us use bilinear filtering to
    // simultaneously evaluate the middle 2 samples from the 4x4 grid.
    float2 w12 = w1 + w2;
    float2 offset12 = w2 / w12;

    // Compute the final UV coordinates we'll use for sampling the texture.
    float2 texPos0 = texPos1 - 1.0f;
    float2 texPos3 = texPos1 + 2.0f;
    float2 texPos12 = texPos1 + offset12;

    texPos0 = texPos0 / texSize;
    texPos3 = texPos3 / texSize;
    texPos12 = texPos12 / texSize;

    T result = ZeroValue<T>();
    result += tex2DLod<T>(texture, texPos0.x, texPos0.y, lod) * (w0.x * w0.y);
    result += tex2DLod<T>(texture, texPos12.x, texPos0.y, lod) * (w12.x * w0.y);
    result += tex2DLod<T>(texture, texPos3.x, texPos0.y, lod) * (w3.x * w0.y);

    result += tex2DLod<T>(texture, texPos0.x, texPos12.y, lod) * (w0.x * w12.y);
    result += tex2DLod<T>(texture, texPos12.x, texPos12.y, lod) * (w12.x * w12.y);
    result += tex2DLod<T>(texture, texPos3.x, texPos12.y, lod) * (w3.x * w12.y);

    result += tex2DLod<T>(texture, texPos0.x, texPos3.y, lod) * (w0.x * w3.y);
    result += tex2DLod<T>(texture, texPos12.x, texPos3.y, lod) * (w12.x * w3.y);
    result += tex2DLod<T>(texture, texPos3.x, texPos3.y, lod) * (w3.x * w3.y);

    return result;
}

template <typename T>
NT_DEVICE inline T
SampleTextureCatmullRom(cudaTextureObject_t texture, const float2 &uv, const float2 &texSize)
{
    return SampleTextureCatmullRomLod<T>(texture, uv, texSize, 0.0f);
}

} // namespace neural_textures
