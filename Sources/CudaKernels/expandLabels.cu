//******************************************************************************
//  Created by Edward Connell
//  Copyright (c) 2016 Connell Research. All rights reserved.
//
#include "include/CudaKernels.h"

//------------------------------------------------------------------------------
// device kernel
template <typename T>
__global__ void expandLabels_kernel(size_t N, const T* labels, size_t labelStride,
																		T *expanded, size_t expNumCols,
																		size_t expRowStride, size_t expColStride)
{
	CUDA_KERNEL_LOOP(i, N) {
		size_t valueIndex = (size_t)labels[i * labelStride];
		expanded[(i * expRowStride) + valueIndex] = 1;
	}
}

//------------------------------------------------------------------------------
// Swift importable C functions
cudaError_t cudaExpandLabels(
	cudaDataType_t dataType,
	const void* labels, size_t labelStride,
	void *expanded, const size_t* expandedExtent, const size_t* expandedStrides,
	cudaStream_t stream)
{
	CudaKernelPreCheck(stream);

	// clear
	size_t count = expandedExtent[0] * expandedStrides[0] * DataTypeSize(dataType);
	cudaMemsetAsync(expanded, 0, count, stream);

	// set label columns to 1
	size_t N = expandedExtent[0];
	switch(dataType) {
		case CUDA_R_32F:
			expandLabels_kernel<float> <<<CUDA_NUM_BLOCKS(N), CUDA_NUM_THREADS, 0, stream>>> (N,
				(float *)labels, labelStride,
				(float *)expanded, expandedExtent[1], expandedStrides[0], expandedStrides[1]);
				break;

		case CUDA_R_64F:
			expandLabels_kernel<double> <<<CUDA_NUM_BLOCKS(N), CUDA_NUM_THREADS, 0, stream>>> (N,
				(double *)labels, labelStride,
				(double *)expanded, expandedExtent[1], expandedStrides[0], expandedStrides[1]);
			break;

		default: assert(false);
	};

	return CudaKernelPostCheck(stream);
}
