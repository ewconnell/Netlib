//******************************************************************************
//  Created by Edward Connell
//  Copyright (c) 2016 Connell Research. All rights reserved.
//
#include "include/CudaKernels.h"
#include <cuda_fp16.h>

#ifdef __JETBRAINS_IDE__
#include "../../../../../../usr/local/cuda/include/driver_types.h"
#include "../../../../../../usr/local/cuda/include/cuda_runtime.h"
#include "../../../../../../usr/local/cuda/include/curand_kernel.h"
#include "../../../../../../usr/lib/gcc/x86_64-linux-gnu/5/include/stddef.h"
#include "../../../../../../usr/include/assert.h"
#include "../../../../../../usr/local/cuda/include/cuda_fp16.h"

#define __CUDACC__ 1
#define __host__
#define __device__
#define __global__
#define __forceinline__
#define __shared__
inline void __syncthreads() {}
inline void __threadfence_block() {}
template<class T> inline T __clz(const T val) { return val; }
struct __cuda_fake_struct { int x; };
extern struct __cuda_fake_struct gridDim;
extern struct __cuda_fake_struct threadIdx;
extern struct __cuda_fake_struct blockIdx;
#endif

//------------------------------------------------------------------------------
// fp16/int
__global__ void softmaxLabelGradient_kernel(
		int N, const half* outData, int xCols, int rowStride,
		const int* labels, float beta, half* inDiff) {

	const half one = __short2half_rd(1);
	half hbeta = __float2half(beta);

	CUDA_KERNEL_LOOP(i, N) {
		int base = i * rowStride;
		for (int j = 0; j < xCols; ++j) {
			inDiff[base + j] = (j == labels[i]) ?
			                   __hmul(hbeta, __hsub(outData[base + j], one)) :
			                   __hmul(hbeta, outData[base + j]);
		}
	}
}

//------------------------------------------------------------------------------
// fp16/float
__global__ void softmaxLabelGradient_kernel(
	int N, const half* outData, int xCols, int rowStride,
	const float* labels, float beta, half* inDiff) {

	const half one = __short2half_rd(1);
	half hbeta = __float2half(beta);

	CUDA_KERNEL_LOOP(i, N) {
		int base = i * rowStride;
		for (int j = 0; j < xCols; ++j) {
			inDiff[base + j] = (j == (int)labels[i]) ?
			                   __hmul(hbeta, __hsub(outData[base + j], one)) :
			                   __hmul(hbeta, outData[base + j]);
		}
	}
}

//------------------------------------------------------------------------------
// T/T
template <typename T, typename L>
__global__ void softmaxLabelGradient_kernel(
		int N, const T* outData, int xCols, int rowStride,
		const L* labels, T beta, T* inDiff) {

	CUDA_KERNEL_LOOP(i, N) {
		int base = i * rowStride;
		for (int j = 0; j < xCols; ++j) {
			inDiff[base + j] = (j == labels[i]) ?
				beta * (outData[base + j] - 1) : beta * outData[base + j];
		}
	}
}

//------------------------------------------------------------------------------
// Swift importable C functions
cudaError_t cudaSoftmaxLabelGradient(
	long N,
	enum cudaDataType_t outDataType,
	const void* outData, long xCols, long rowStride,
	enum cudaDataType_t labelsDataType,
	const int* labels,
	double beta,
	enum cudaDataType_t inDiffDataType,
	void* inDiff,
	cudaStream_t stream) {

	// validate
	assert(outDataType == inDiffDataType);
	if (labelsDataType != CUDA_R_32I && labelsDataType != CUDA_R_32F) {
		return cudaErrorNotSupported;
	}

	CudaKernelPreCheck(stream);

	if (labelsDataType == CUDA_R_32I) {
		switch(outDataType) {
			case CUDA_R_16F:
					softmaxLabelGradient_kernel<<<CUDA_NUM_BLOCKS(N), CUDA_NUM_THREADS, 0, stream>>>
						(N, (half *)outData, xCols, rowStride, labels, (float)beta, (half *)inDiff);
				break;

			case CUDA_R_32F:
				softmaxLabelGradient_kernel<float> <<<CUDA_NUM_BLOCKS(N), CUDA_NUM_THREADS, 0, stream>>>
					(N, (float *)outData, xCols, rowStride, labels, (float)beta, (float *)inDiff);
				break;

			case CUDA_R_64F:
				softmaxLabelGradient_kernel<double> <<<CUDA_NUM_BLOCKS(N), CUDA_NUM_THREADS, 0, stream>>>
					(N, (double *)outData, xCols, rowStride, labels, beta, (double *)inDiff);
				break;

			default: assert(false);
		};
	} else {
		switch(outDataType) {
			case CUDA_R_16F:
					softmaxLabelGradient_kernel<<<CUDA_NUM_BLOCKS(N), CUDA_NUM_THREADS, 0, stream>>>
						(N, (half *)outData, xCols, rowStride, (float*)labels, (float)beta, (half *)inDiff);
				break;

			case CUDA_R_32F:
				softmaxLabelGradient_kernel<float> <<<CUDA_NUM_BLOCKS(N), CUDA_NUM_THREADS, 0, stream>>>
					(N, (float *)outData, xCols, rowStride, (float*)labels, (float)beta, (float *)inDiff);
				break;

			case CUDA_R_64F:
				softmaxLabelGradient_kernel<double> <<<CUDA_NUM_BLOCKS(N), CUDA_NUM_THREADS, 0, stream>>>
					(N, (double *)outData, xCols, rowStride, (float*)labels, beta, (double *)inDiff);
				break;

			default: assert(false);
		};
	}

	return CudaKernelPostCheck(stream);
}
