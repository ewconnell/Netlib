//******************************************************************************
//  Created by Edward Connell
//  Copyright (c) 2016 Connell Research. All rights reserved.
//
#include "include/CudaKernels.h"

#ifdef __JETBRAINS_IDE__
#include "../../../../../../usr/local/cuda/include/driver_types.h"
#include "../../../../../../usr/local/cuda/include/cuda_runtime.h"
#include "../../../../../../usr/local/cuda/include/cuda_fp16.h"
#include "../../../../../../usr/include/assert.h"
#include "../../../../../../usr/local/cuda/include/cublas_v2.h"

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
// device kernel
__global__ void updateGradient_kernel(unsigned count, half* weights, const half* gradient, float rate) {
//	half alpha = __float2half(falpha);
//	half beta = __float2half(fbeta);
//
//	CUDA_KERNEL_LOOP(i, count) {
//		x[i] = y[i] = __hadd(__hmul(alpha, x[i]), __hmul(beta, y[i]));
//	}
}

template <typename T>
__global__ void updateGradient_kernel(unsigned count, T* weights, const T* gradient, T rate) {
	CUDA_KERNEL_LOOP(i, count) {
        weights[i] -= gradient[i] * rate;
	}
}


__global__ void updateGradient_kernel(unsigned count, half* weights, const half* gradient, float frate,
                                      half* history, float fmomentum) {
	half rate = __float2half(frate);
	half momentum = __float2half(fmomentum);

	CUDA_KERNEL_LOOP(i, count) {
		history[i] = __hadd(__hmul(gradient[i], rate), __hmul(history[i], momentum));
        weights[i] = __hsub(weights[i], history[i]);
	}
}

template <typename T>
__global__ void updateGradient_kernel(unsigned count, T* weights, const T* gradient, T rate,
									  T* history, T momentum) {
	CUDA_KERNEL_LOOP(i, count) {
        history[i] = gradient[i] * rate + history[i] * momentum;
		weights[i] -= history[i];
	}
}

//------------------------------------------------------------------------------
// Swift importable C functions
cudaError_t cudaUpdateGradient(enum cudaDataType_t dataType, long count,
							   void *weights, const void *gradient,
                               double learningRate, cudaStream_t stream) {
	CudaKernelPreCheck(stream);

	switch(dataType) {
		case CUDA_R_16F:
            updateGradient_kernel<<<CUDA_NUM_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>
                    (count, (half *)weights, (const half *)gradient, (float)learningRate);
			break;

		case CUDA_R_32F:
            updateGradient_kernel<float> <<<CUDA_NUM_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>
				(count, (float *)weights, (const float *)gradient, (float)learningRate);
			break;

		case CUDA_R_64F:
            updateGradient_kernel<double> <<<CUDA_NUM_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>
                    (count, (double *)weights, (const double *)gradient, (double)learningRate);
			break;

		default: assert(false);
	};
	return CudaKernelPostCheck(stream);
}

cudaError_t cudaUpdateGradientWithMomentum(
		enum cudaDataType_t dataType, long count,
		void *weights, const void *gradient, double learningRate,
		void *history, double momentum,
		cudaStream_t stream) {
	CudaKernelPreCheck(stream);

	switch(dataType) {
		case CUDA_R_16F:
			updateGradient_kernel<<<CUDA_NUM_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>
					(count, (half *)weights, (const half *)gradient, (float)learningRate,
							(half*)history, (float)momentum);
			break;

		case CUDA_R_32F:
			updateGradient_kernel<float> <<<CUDA_NUM_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>
					(count, (float *)weights, (const float *)gradient, (float)learningRate,
							(float *)history, (float)momentum);
			break;

		case CUDA_R_64F:
			updateGradient_kernel<double> <<<CUDA_NUM_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>
					(count, (double *)weights, (const double *)gradient, learningRate,
							(double *)history, momentum);
			break;

		default: assert(false);
	};
	return CudaKernelPostCheck(stream);
}
