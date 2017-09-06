//******************************************************************************
//  Created by Edward Connell on 4/4/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
import Cuda
import Foundation

//==============================================================================
// CudaComputeService
public final class CudaComputeService : ComputeService {
	// initializers
	public required init(log: Log?) throws {
		currentLog = log

		// this is a singleton
		trackingId = objectTracker.register(type: self)
		objectTracker.markStatic(trackingId: trackingId)

		// create devices
		var deviceCount: CInt = 0
		do {
			try cudaCheck(status: cudaGetDeviceCount(&deviceCount))
		} catch {
			writeLog(
				"cudaGetDeviceCount failed. The Cuda driver may be in an unstable state",
				level: .error)
			throw error
		}

		guard deviceCount > 0 else {
			writeLog("There are no '\(name)' devices installed", level: .warning)
			throw ComputeError.serviceIsUnavailable
		}

		// add device object for each id reported
		for i in 0..<Int(deviceCount) {
			try devices.append(CudaDevice(log: log, service: self, id: i))
		}
	}

	deinit { objectTracker.remove(trackingId: trackingId) }

	//----------------------------------------------------------------------------
	// properties
	public private(set) var trackingId = 0
	public let name = "cuda"
	public var id = 0
	public var devices = [ComputeDevice]()

	// logging
	public var logLevel = LogLevel.error
	public var nestingLevel = 0
	public weak var currentLog: Log?
}

//==============================================================================
// cudaCheck cudaError_t
public func cudaCheck(status: cudaError_t, file: String = #file,
                      function: String = #function, line: Int = #line) throws {
	if status != cudaSuccess {
		let location = "CUDA error in \(file) at \(function):\(line)"
		let message = String(utf8String: cudaGetErrorString(status))!
		cudaDeviceReset()
		throw ComputeError.functionFailure(location: location, message: message)
	}
}

//==============================================================================
// cudaCheck cudnnStatus_t
public func cudaCheck(status: cudnnStatus_t, file: String = #file,
                      function: String = #function, line: Int = #line) throws {
	if status != CUDNN_STATUS_SUCCESS {
		let location = "CUDNN error in \(file) at \(function):\(line)"
		let message = String(utf8String: cudnnGetErrorString(status))!
		print(message)
		cudaDeviceReset()
		throw ComputeError.functionFailure(location: location, message: message)
	}
}

//==============================================================================
// cudaCheck cublasStatus_t
public func cudaCheck(status: cublasStatus_t, file: String = #file,
                      function: String = #function, line: Int = #line) throws {
	if status != CUBLAS_STATUS_SUCCESS {
		let location = "CUBLAS error in \(file) at \(function):\(line)"
		let message = String(utf8String: cublasGetErrorString(status))! + "code=(\(status))"
		cudaDeviceReset()
		throw ComputeError.functionFailure(location: location, message: message)
	}
}

public func cublasGetErrorString(_ status: cublasStatus_t) -> String {
	switch status {
	case CUBLAS_STATUS_SUCCESS         : return "CUBLAS_STATUS_SUCCESS"
	case CUBLAS_STATUS_NOT_INITIALIZED : return "CUBLAS_STATUS_NOT_INITIALIZED"
	case CUBLAS_STATUS_ALLOC_FAILED    : return "CUBLAS_STATUS_ALLOC_FAILED"
	case CUBLAS_STATUS_INVALID_VALUE   : return "CUBLAS_STATUS_INVALID_VALUE"
	case CUBLAS_STATUS_ARCH_MISMATCH   : return "CUBLAS_STATUS_ARCH_MISMATCH"
	case CUBLAS_STATUS_MAPPING_ERROR   : return "CUBLAS_STATUS_MAPPING_ERROR"
	case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED"
	case CUBLAS_STATUS_INTERNAL_ERROR  : return "CUBLAS_STATUS_INTERNAL_ERROR"
	case CUBLAS_STATUS_NOT_SUPPORTED   : return "CUBLAS_STATUS_NOT_SUPPORTED"
	case CUBLAS_STATUS_LICENSE_ERROR   : return "CUBLAS_STATUS_LICENSE_ERROR"
	default: return "<unknown>"
	}
}

//==============================================================================
// cudaCheck curandStatus_t
public func cudaCheck(status: curandStatus_t, file: String = #file,
                      function: String = #function, line: Int = #line) throws {
	if status != CURAND_STATUS_SUCCESS {
		let location = "CURAND error in \(file) at \(function):\(line)"
		let message = String(utf8String: curandGetErrorString(status))! + "code=(\(status))"
		cudaDeviceReset()
		throw ComputeError.functionFailure(location: location, message: message)
	}
}

public func curandGetErrorString(_ status: curandStatus_t) -> String {
	switch status {
	case CURAND_STATUS_SUCCESS:	return "CURAND_STATUS_SUCCESS"
	case CURAND_STATUS_VERSION_MISMATCH:	return "CURAND_STATUS_VERSION_MISMATCH"
	case CURAND_STATUS_NOT_INITIALIZED:	return "CURAND_STATUS_NOT_INITIALIZED"
	case CURAND_STATUS_ALLOCATION_FAILED: return "CURAND_STATUS_ALLOCATION_FAILED"
	case CURAND_STATUS_TYPE_ERROR: return "CURAND_STATUS_TYPE_ERROR"
	case CURAND_STATUS_OUT_OF_RANGE: return "CURAND_STATUS_OUT_OF_RANGE"
	case CURAND_STATUS_LENGTH_NOT_MULTIPLE: return "CURAND_STATUS_LENGTH_NOT_MULTIPLE"
	case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED: return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED"
	case CURAND_STATUS_LAUNCH_FAILURE: return "CURAND_STATUS_LAUNCH_FAILURE"
	case CURAND_STATUS_PREEXISTING_FAILURE: return "CURAND_STATUS_PREEXISTING_FAILURE"
	case CURAND_STATUS_INITIALIZATION_FAILED: return "CURAND_STATUS_INITIALIZATION_FAILED"
	case CURAND_STATUS_ARCH_MISMATCH: return "CURAND_STATUS_ARCH_MISMATCH"
	case CURAND_STATUS_INTERNAL_ERROR: return "CURAND_STATUS_INTERNAL_ERROR"
	default: return "<unknown>"
	}
}

//==============================================================================
// Conversion extensions
extension DataLayout {
	public var cudnn: cudnnTensorFormat_t {
		switch self {
		case .nhwc: return CUDNN_TENSOR_NHWC
		case .nchw_vector_c: return CUDNN_TENSOR_NCHW_VECT_C
		case .nchw, .vector, .matrix: return CUDNN_TENSOR_NCHW
		}
	}
}

//------------------------------------------------------------------------------
// TransposeOp
extension TransposeOp {
	public var cublas: cublasOperation_t {
		switch self {
		case .noTranspose: return CUBLAS_OP_N
		case .transpose: return CUBLAS_OP_T
		case .conjugateTranspose: return CUBLAS_OP_C
		}
	}
}

//------------------------------------------------------------------------------
// ReductionOp extension
extension ReductionOp {
	public var cudnn: cudnnReduceTensorOp_t {
		get {
			switch self {
			case .add: return CUDNN_REDUCE_TENSOR_ADD
			case .mul: return CUDNN_REDUCE_TENSOR_MUL
			case .min: return CUDNN_REDUCE_TENSOR_MIN
			case .max: return CUDNN_REDUCE_TENSOR_MAX
			case .amax: return CUDNN_REDUCE_TENSOR_AMAX
			case .avg: return CUDNN_REDUCE_TENSOR_AVG
			case .norm1: return CUDNN_REDUCE_TENSOR_NORM1
			case .norm2: return CUDNN_REDUCE_TENSOR_NORM2
			}
		}
	}
}

//------------------------------------------------------------------------------
// DataType extension
extension DataType {
	public var cudnn: cudnnDataType_t {
		get {
			switch self {
			case .real8U: return CUDNN_DATA_INT8
			case .real32I: return CUDNN_DATA_INT32
			case .real16F: return CUDNN_DATA_HALF
			case .real32F: return CUDNN_DATA_FLOAT
			case .real64F: return CUDNN_DATA_DOUBLE
			default: fatalError("Invalid state")
			}
		}
	}

	public var cuda: cudaDataType_t {
		get {
			switch self {
			case .real16F: return CUDA_R_16F
			case .real32F: return CUDA_R_32F
			case .real64F: return CUDA_R_64F
			case .real8U : return CUDA_R_8U
			case .real32I: return CUDA_R_32I
			default: fatalError("not supported")
			}
		}
	}
}

//------------------------------------------------------------------------------
// NanPropagation
extension NanPropagation {
	public var cudnn: cudnnNanPropagation_t {
		get {
			switch self {
			case .noPropagate: return CUDNN_NOT_PROPAGATE_NAN
			case .propagate: return CUDNN_PROPAGATE_NAN
			}
		}
	}
}

//==============================================================================
// CudnnHandle
public final class CudnnHandle : ObjectTracking {
	init(deviceId: Int, using stream: cudaStream_t) throws {
		self.deviceId = deviceId
		try cudaCheck(status: cudaSetDevice(Int32(deviceId)))

		var temp: cudnnHandle_t?
		try cudaCheck(status: cudnnCreate(&temp))
		handle = temp!
		try cudaCheck(status: cudnnSetStream(handle, stream))
		trackingId = objectTracker.register(type: self)
	}

	deinit {
		do {
			try cudaCheck(status: cudaSetDevice(Int32(deviceId)))
			try cudaCheck(status: cudnnDestroy(handle))
			objectTracker.remove(trackingId: trackingId)
		} catch {
			print("\(releaseString) CudnnHandle(\(trackingId)) \(String(describing: error))")
		}
	}

	// properties
	public private (set) var trackingId = 0
	private let deviceId: Int
	public var handle: cudnnHandle_t
}

//==============================================================================
// CublasHandle
public final class CublasHandle : ObjectTracking {
	public init(deviceId: Int, using stream: cudaStream_t) throws {
		self.deviceId = deviceId
		try cudaCheck(status: cudaSetDevice(Int32(deviceId)))

		var temp: cublasHandle_t?
		try cudaCheck(status: cublasCreate_v2(&temp))
		handle = temp!
		try cudaCheck(status: cublasSetStream_v2(handle, stream))
		trackingId = objectTracker.register(type: self)
	}

	deinit {
		do {
			try cudaCheck(status: cudaSetDevice(Int32(deviceId)))
			try cudaCheck(status: cublasDestroy_v2(handle))
			objectTracker.remove(trackingId: trackingId)
		} catch {
			print(String(describing: error))
		}
	}

	// properties
	public private (set) var trackingId = 0
	private let deviceId: Int
	public var handle: cublasHandle_t
}

//==============================================================================
// ActivationDescriptor
public final class ActivationDescriptor : ObjectTracking {
	public init(mode: ActivationMode, nan: NanPropagation, reluCeiling: Double) throws {
		// create the descriptor
		var temp: cudnnActivationDescriptor_t?
		try cudaCheck(status: cudnnCreateActivationDescriptor(&temp))
		desc = temp!
		
		// initialize
		try cudaCheck(status: cudnnSetActivationDescriptor(
			desc, mode.cudnn, nan.cudnn, reluCeiling))
		trackingId = objectTracker.register(type: self)
	}

	deinit {
		try! cudaCheck(status: cudnnDestroyActivationDescriptor(desc))
		objectTracker.remove(trackingId: trackingId)
	}

	// properties
	public private (set) var trackingId = 0
	let desc: cudnnActivationDescriptor_t
}

//==============================================================================
// ConvolutionDescriptor
public final class ConvolutionDescriptor : ObjectTracking {
	// initializers
	public init(dataType: DataType, rank: Int, pad: [Int],
	            stride: [Int], dilation: [Int], mode: ConvolutionMode) throws {
		// create the descriptor
		var temp: cudnnConvolutionDescriptor_t?
		try cudaCheck(status: cudnnCreateConvolutionDescriptor(&temp))
		desc = temp!
		
		// initialize
		try cudaCheck(status: cudnnSetConvolutionNdDescriptor(
			desc, CInt(rank),
			pad.map { CInt($0) },
			stride.map { CInt($0) },
			dilation.map { CInt($0) },
			mode.cudnn,
			dataType.cudnn))

		trackingId = objectTracker.register(type: self)
	}

	deinit {
		try! cudaCheck(status: cudnnDestroyConvolutionDescriptor(desc))
		objectTracker.remove(trackingId: trackingId)
	}
	
	// properties
	public private (set) var trackingId = 0
	let desc: cudnnConvolutionDescriptor_t
}


//==============================================================================
// DropoutDescriptor
public final class DropoutDescriptor : ObjectTracking {
	// initializers
	public init(stream: CudaStream, drop: Double, seed: UInt64,
	            tensorDesc: TensorDescriptor) throws {
		// create the descriptor
		var temp: cudnnDropoutDescriptor_t?
		try cudaCheck(status: cudnnCreateDropoutDescriptor(&temp))
		desc = temp!

		// get states size
		var stateSizeInBytes = 0
		try cudaCheck(status: cudnnDropoutGetStatesSize(
			tensorDesc.desc, &stateSizeInBytes))

		// create states array
		states = try stream.device.createArray(count: stateSizeInBytes)

		// initialize
		try cudaCheck(status: cudnnSetDropoutDescriptor(
			desc,
			stream.cudnn.handle,
			Float(drop),
			states.data,
			states.count,
			seed
		))

		trackingId = objectTracker.register(type: self)
	}

	deinit {
		try! cudaCheck(status: cudnnDestroyDropoutDescriptor(desc))
		objectTracker.remove(trackingId: trackingId)
	}

	// properties
	private var states: DeviceArray
	public private (set) var trackingId = 0
	let desc: cudnnDropoutDescriptor_t
}

//==============================================================================
// FilterDescriptor
public final class FilterDescriptor : ObjectTracking {
	// initializers
	public init(shape: Shape, dataType: DataType) throws {
		// create the descriptor
		var temp: cudnnFilterDescriptor_t?
		try cudaCheck(status: cudnnCreateFilterDescriptor(&temp))
		desc = temp!
		
		// initialize
		try cudaCheck(status: cudnnSetFilterNdDescriptor(
			desc, dataType.cudnn,
			shape.layout.cudnn,
			Int32(shape.extent.count),
			shape.extent.map { Int32($0)}))

		trackingId = objectTracker.register(type: self)
	}

	deinit {
		try! cudaCheck(status: cudnnDestroyFilterDescriptor(desc))
		objectTracker.remove(trackingId: trackingId)
	}
	
	// properties
	public private (set) var trackingId = 0
	let desc: cudnnFilterDescriptor_t
}

//==============================================================================
// LRNDescriptor
public final class LRNDescriptor : ObjectTracking {
	// initializers
	public init(N: Int, alpha: Double, beta: Double, K: Double) throws {
		guard N >= Int(CUDNN_LRN_MIN_N) && N <= Int(CUDNN_LRN_MAX_N) else {
			throw ModelError.rangeError(
				"N = \(N) is invalid. Range \(CUDNN_LRN_MIN_N) to \(CUDNN_LRN_MAX_N)")
		}
		guard K >= CUDNN_LRN_MIN_K else {
			throw ModelError.rangeError(
				"K = \(K) is invalid. Must be >= to \(CUDNN_LRN_MIN_K)")
		}
		guard beta >= CUDNN_LRN_MIN_BETA else {
			throw ModelError.rangeError(
				"beta = \(beta) is invalid. Must be >= to \(CUDNN_LRN_MIN_BETA)")
		}
		
		// create the descriptor
		var temp: cudnnLRNDescriptor_t?
		try cudaCheck(status: cudnnCreateLRNDescriptor(&temp))
		desc = temp!
		
		// initialize
		try cudaCheck(status: cudnnSetLRNDescriptor(desc, CUnsignedInt(N), alpha, beta, K))
		trackingId = objectTracker.register(type: self)
	}

	deinit {
		try! cudaCheck(status: cudnnDestroyLRNDescriptor(desc))
		objectTracker.remove(trackingId: trackingId)
	}
	
	// properties
	public private (set) var trackingId = 0
	let desc: cudnnLRNDescriptor_t
}

//==============================================================================
// TensorDescriptor
public final class TensorDescriptor : ObjectTracking {
	// initializers
	public init(shape: Shape, dataType: DataType) throws {
		// create the descriptor
		var temp: cudnnTensorDescriptor_t?
		try cudaCheck(status: cudnnCreateTensorDescriptor(&temp))
		desc = temp!
		
		// initialize
		try cudaCheck(status: cudnnSetTensorNdDescriptor(
			desc, dataType.cudnn,
			Int32(shape.extent.count),
			shape.extent.map { Int32($0)},
			shape.strides.map { Int32($0)}))

		trackingId = objectTracker.register(type: self)
	}

	public init(owning desc: cudnnTensorDescriptor_t) {
		self.desc = desc
		trackingId = objectTracker.register(type: self)
	}

	deinit {
		try! cudaCheck(status: cudnnDestroyTensorDescriptor(desc))
		objectTracker.remove(trackingId: trackingId)
	}
	
	// properties
	public private (set) var trackingId = 0
	let desc: cudnnTensorDescriptor_t
}

//==============================================================================
// createTensorDescriptor
extension DataView {
	public func createTensorDescriptor(asShape newShape: Shape? = nil) throws -> TensorDescriptor {
		assert(newShape == nil || newShape!.elementCount == shape.elementCount)
		return try TensorDescriptor(shape: newShape ?? shape, dataType: dataType)
	}
}

//==============================================================================
// PoolingDescriptor
public final class PoolingDescriptor : ObjectTracking {
	// initializers
	public init(mode: PoolingMode, nan: NanPropagation, rank: Int, window: [Int],
	            padding: [Int], stride: [Int]) throws {
		// create the descriptor
		var temp: cudnnPoolingDescriptor_t?
		try cudaCheck(status: cudnnCreatePoolingDescriptor(&temp))
		desc = temp!
		
		// initialize
		try cudaCheck(status: cudnnSetPoolingNdDescriptor(
			desc, mode.cudnn, nan.cudnn,
			CInt(rank),
			window.map { CInt($0) },
			padding.map { CInt($0) },
			stride.map { CInt($0) }))

		trackingId = objectTracker.register(type: self)
	}

	deinit {
		try! cudaCheck(status: cudnnDestroyLRNDescriptor(desc))
		objectTracker.remove(trackingId: trackingId)
	}
	
	// properties
	public private (set) var trackingId = 0
	let desc: cudnnPoolingDescriptor_t
}

//==============================================================================
// ReduceTensorDescriptor
public final class ReduceTensorDescriptor : ObjectTracking {
	// initializers
	public init(op: ReductionOp, nan: NanPropagation, dataType: DataType) throws {
		// create the descriptor
		var temp: cudnnReduceTensorDescriptor_t?
		try cudaCheck(status: cudnnCreateReduceTensorDescriptor(&temp))
		desc = temp!

		let indicesAction = (op == .min || op == .max) ?
			CUDNN_REDUCE_TENSOR_FLATTENED_INDICES : CUDNN_REDUCE_TENSOR_NO_INDICES

		// initialize
		try cudaCheck(status: cudnnSetReduceTensorDescriptor(
			desc,
			op.cudnn,
			dataType == .real64F ? CUDNN_DATA_DOUBLE : CUDNN_DATA_FLOAT,
			nan.cudnn,
			indicesAction,
			CUDNN_32BIT_INDICES
		))

		trackingId = objectTracker.register(type: self)
	}

	deinit {
		try! cudaCheck(status: cudnnDestroyReduceTensorDescriptor(desc))
		objectTracker.remove(trackingId: trackingId)
	}

	// properties
	public private (set) var trackingId = 0
	let desc: cudnnReduceTensorDescriptor_t
}







