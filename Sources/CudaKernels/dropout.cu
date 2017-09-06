//******************************************************************************
//  Created by Edward Connell
//  Copyright (c) 2016 Connell Research. All rights reserved.
//
#include "include/CudaKernels.h"
#include <curand_kernel.h>

//------------------------------------------------------------------------------
// dropoutForward_kernel
template <typename T>
__global__ void dropoutForward_kernel(
	curandState *state, T threshold, T scale, T* mask,
	const cudaShape_t inShape, const T* inData,
	const cudaShape_t outShape, T* outData)
{
	CUDA_KERNEL_LOOP(i, inShape.extent[0]) {
		unsigned iIdx = i * inShape.stride[0];
		unsigned oIdx = i * outShape.stride[0];
		mask[i] = (curand_uniform(&state[i]) > threshold) ? scale : 0;

		// multiply by either 0 or the scale
		outData[oIdx] = inData[iIdx] * mask[i];
	}
}

//-------------------------------------
// cudaDropoutForward
cudaError_t cudaDropoutForward(
	const cudaShape_t inShape, const void *inData,
	const cudaShape_t outShape, void *outData,
	double ratio, void *mask,
	void* generatorState, cudaStream_t stream)
{
	// require flattening for now
	assert(inShape.dataType == outShape.dataType);

	unsigned numBlocks  = CUDA_NUM_BLOCKS(inShape.extent[0]);
	unsigned numThreads = CUDA_NUM_THREADS;

	double scale = 1.0 / (1.0 - ratio);

	switch(inShape.dataType) {
		case CUDA_R_32F:
			dropoutForward_kernel<float> <<<numBlocks, numThreads, 0, stream>>>
				((curandState *) generatorState, (float)ratio, (float)scale, (float*)mask,
					inShape, (float*)inData, outShape, (float*)outData);
			break;

		case CUDA_R_64F:
			dropoutForward_kernel<double> <<<numBlocks, numThreads, 0, stream>>>
			((curandState *) generatorState, ratio, scale, (double*)mask,
				inShape, (double*)inData, outShape, (double*)outData);
			break;

		default: assert(false);
	};

	return CudaKernelPostCheck(stream);
}

//------------------------------------------------------------------------------
// dropoutBackward_kernel
template <typename T>
__global__ void dropoutBackward_kernel(
	const cudaShape_t outShape, const T* outDiff,
	const cudaShape_t inShape, T* inDiff, const T* mask)
{
	CUDA_KERNEL_LOOP(i, inShape.extent[0]) {
		unsigned iIdx = i * inShape.stride[0];
		unsigned oIdx = i * outShape.stride[0];

		inDiff[iIdx] = outDiff[oIdx] * mask[i];
	}
}

//-------------------------------------
// cudaDropoutBackward
cudaError_t cudaDropoutBackward(
	const cudaShape_t outShape, const void *outDiff,
	const cudaShape_t inShape, void *inDiff,
	const void *mask, cudaStream_t stream)
{
	CudaKernelPreCheck(stream);

	// require flattening for now
	assert(inShape.dataType == outShape.dataType);

	unsigned numBlocks  = CUDA_NUM_BLOCKS(inShape.extent[0]);
	unsigned numThreads = CUDA_NUM_THREADS;

	switch(inShape.dataType) {
		case CUDA_R_32F:
			dropoutBackward_kernel<float> <<<numBlocks, numThreads, 0, stream>>>
			(outShape, (float*)outDiff, inShape, (float*)inDiff, (float*)mask);
			break;

		case CUDA_R_64F:
			dropoutBackward_kernel<double> <<<numBlocks, numThreads, 0, stream>>>
			(outShape, (double*)outDiff, inShape, (double*)inDiff, (double*)mask);
			break;

		default: assert(false);
	};

	return CudaKernelPostCheck(stream);
}
