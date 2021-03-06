#pragma once

// This file is used for stubbing out the GPUTexture interface when compiled without CUDA.

#ifndef NO_CUDA

#error "This header should only be used if NO_CUDA is defined"

#endif

#include <string>

struct GPUTexture
{
	static GPUTexture fromPng(const std::string& path) {}
};
