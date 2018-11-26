//******************************************************************************
//  Created by Edward Connell
//  Copyright (c) 2016 Connell Research. All rights reserved.
//
#include "include/CudaKernels.h"
//#include "../../../../../../usr/local/cuda/include/host_defines.h"
//#include "../../../../../../usr/local/cuda/include/device_launch_parameters.h"
//#include "../../../../../../usr/local/cuda/include/cuda_runtime.h"
//#include "../../../../../../usr/include/assert.h"

//------------------------------------------------------------------------------
// device kernel
__device__ int64_t globalElapsed;

__global__ void cudaDelayStream_kernel(int64_t count)
{
	clock_t start = clock64();
	clock_t elapsed = 0;
	while(elapsed < count) {
		elapsed = clock64() - start;
	}
	globalElapsed = elapsed;
}

//------------------------------------------------------------------------------
// Swift importable C functions
cudaError_t cudaDelayStream(double seconds, int clockRate, cudaStream_t stream)
{
	CudaKernelPreCheck(stream);

	int64_t count = seconds * clockRate * 1000;
	cudaDelayStream_kernel<<<1, 1, 0, stream>>>(count);
	return CudaKernelPostCheck(stream);
}
