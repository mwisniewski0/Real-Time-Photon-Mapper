#include "gpuTextures.h"

/*
 * Creates a texture from a png image and copies data to gpu.
 */
GPUTexture GPUTexture::fromPng(const std::string& path)
{
	std::vector<unsigned char> image; //the raw pixels
	unsigned width, height;

	unsigned error = lodepng::decode(image, width, height, path.c_str());

	if (error)
	{
		throw std::runtime_error(lodepng_error_text(error));
	}

	GPUTexture result;
	result.width = width;
	result.height = height;
	result.colors = vectorToGpu(image);
	return result;
}
