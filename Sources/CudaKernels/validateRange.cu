//******************************************************************************
//  Created by Edward Connell
//  Copyright (c) 2016 Connell Research. All rights reserved.
//
#include "include/CudaKernels.h"
#include "../../../../../../usr/local/cuda/include/cuda_runtime.h"
#include "../../../../../../usr/include/assert.h"
#include "../../../../../../usr/include/math.h"
#include "../../../../../../usr/local/cuda/include/device_launch_parameters.h"
#include "../../../../../../usr/local/cuda/include/cuda.h"

//------------------------------------------------------------------------------
// device kernel
template <typename T>
__global__ void validateRange_kernel1(const cudaShape_t inShape, const T* inData,
																			const cudaShape_t outShape, T* outData)
{
	CUDA_KERNEL_LOOP(i, inShape.extent[0]) {
		// TODO: this is probably divergent, think of a better way
		if(!isfinite(inData[i * inShape.stride[0]])) outData[0] = 1;
	}
}

//------------------------------------------------------------------------------
// Swift importable C functions
//	returns
//		0 all values fall within range
//		1 one or more values are out of range
//
cudaError_t cudaValidateRange(const cudaShape_t inShape, const void *inData,
							  const cudaShape_t outShape, void *outData,
							  cudaStream_t stream)
{
	CudaKernelPreCheck(stream);

	// require flattening for now
	assert(inShape.dataType == outShape.dataType);

	unsigned numBlocks = CUDA_NUM_BLOCKS(inShape.extent[0]);
	unsigned numThreads = CUDA_NUM_THREADS;
	cudaError_t status;

	switch(inShape.dataType) {
		case CUDA_R_32F:
			status = cudaMemsetAsync(outData, 0, sizeof(float), stream);

			validateRange_kernel1<float> <<<numBlocks, numThreads, 0, stream>>>
				(inShape, (float*)inData, outShape, (float*)outData);
			break;

		case CUDA_R_64F:
			status = cudaMemsetAsync(outData, 0, sizeof(double), stream);

			validateRange_kernel1<double> <<<numBlocks, numThreads, 0, stream>>>
				(inShape, (double*)inData, outShape, (double*)outData);
			break;

		default: assert(false);
	};

	return status != cudaSuccess ? status : CudaKernelPostCheck(stream);
}
