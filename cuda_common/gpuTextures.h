#pragma once
#include <device_launch_parameters.h>
#include "../cuda_common/cudaHelpers.h"
#include "../external/lodepng.h"

/*
 * Stores a texture on the gpu.
 */
struct GPUTexture
{
	GPUVector<unsigned char> colors;
	unsigned width;
	unsigned height;

    /*
     * Returns the color of the pixel in column x and row y of image.
     */
	__device__ float3 getPixelColor(const unsigned& x, const unsigned& y)
	{
		float3 result;
		result.x = colors.contents[4 * (x + y * width) + 0] / 255.f;
		result.y = colors.contents[4 * (x + y * width) + 1] / 255.f;
		result.z = colors.contents[4 * (x + y * width) + 2] / 255.f;
		return result;
	}

    /*
     * Returns color at corresponding texel (x and y range from 0 to 1).
     */
	__device__ float3 getTexelColor(const float& x, const float& y)
	{		
		unsigned x_i = unsigned(x * width);
		unsigned y_i = unsigned(y * height);

		if (x_i >= width)
		{
			x_i = width - 1;
		}
		if (y_i >= height)
		{
			y_i = height - 1;
		}

		float3 result;
		result.x = colors.contents[4 * (x_i + y_i * width) + 0] / 255.f;
		result.y = colors.contents[4 * (x_i + y_i * width) + 1] / 255.f;
		result.z = colors.contents[4 * (x_i + y_i * width) + 2] / 255.f;
		return result;
	}

	// Builds a texture using the PNG located at the provided file path.
	static GPUTexture fromPng(const std::string& path);
};
