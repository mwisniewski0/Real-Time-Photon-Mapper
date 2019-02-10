#pragma once
#include <device_launch_parameters.h>
#include "../cuda_common/cudaHelpers.h"
#include "../external/lodepng.h"

struct GPUTexture
{
	GPUVector<unsigned char> colors;
	unsigned width;
	unsigned height;

	__device__ float3 getPixelColor(const unsigned& x, const unsigned& y)
	{
		float3 result;
		result.x = colors.contents[4 * (x + y * width) + 0] / 255.f;
		result.y = colors.contents[4 * (x + y * width) + 1] / 255.f;
		result.z = colors.contents[4 * (x + y * width) + 2] / 255.f;
		return result;
	}

	__device__ float3 getTexelColor(const float& x, const float& y)
	{

		// unsigned x_i = __float2uint_rn(x * width);
		// unsigned y_i = __float2uint_rn(y * height);
		
		unsigned x_i = unsigned(x * width);
		unsigned y_i = unsigned(y * height);

		// printf("%d %d, %d, %d\n", x_i, y_i, 4 * (x_i + y_i * width), colors.size);
		// return make_float3(0.0f, 0.0f, 0.0f);

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

	static GPUTexture fromPng(const std::string& path);
};
