#pragma once

// This file defines various helpers used throughout Photon's CUDA code

#include <stdexcept>
#include <cuda_runtime.h>
#include <vector>

#define CUDA_CHECK(call) { checkCudaError((call), __FILE__, __LINE__); }

/*
 * Check a cuda error code. If there is an error, print it and exit program.
 */
inline void checkCudaError(cudaError_t code, bool abort = false)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s\n", cudaGetErrorString(code));
		if (abort) exit(code);
	}
}

/*
 * std::vector doesn't work well with cuda. So here is our own simple implemetation that does work.
 */
template <typename T>
struct GPUVector
{
	T* contents;
	unsigned size;

	void release()
	{
		cudaFree(contents);
	}
	
    __device__ T& operator[](unsigned idx)
	{
		return contents[idx];
    }
};

/*
 * Converts a vector on the cpu into a vector on the gpu.
 */
template <typename T>
GPUVector<T> vectorToGpu(const std::vector<T>& v)
{
	GPUVector<T> result;
	result.size = v.size();
	checkCudaError(cudaMalloc(&result.contents, v.size() * sizeof(T)));
	checkCudaError(cudaMemcpy(result.contents, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
	return result;
}
