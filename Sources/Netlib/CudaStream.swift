//******************************************************************************
//  Created by Edward Connell on 4/5/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
import Cuda
import CudaKernels

public final class CudaStream : DeviceStream {
	// initializers
	public init(log: Log?, device: CudaDevice, id: Int, label: String) throws {
		self.device = device
		self.id     = id
		self.label  = label
		currentLog  = log

		// select the specified device
		try device.select()

		// create a stream associated with the device
		var cudaStream: cudaStream_t?
		try cudaCheck(status:
			cudaStreamCreateWithFlags(&cudaStream, UInt32(cudaStreamNonBlocking)))

		handle = cudaStream!
		cudnn  = try CudnnHandle(deviceId: device.id, using: handle)
		cublas = try CublasHandle(deviceId: device.id, using: handle)
		trackingId = objectTracker.register(type: self)

		if willLog(level: .diagnostic) {
			diagnostic("\(createString) \(label)", categories: .streamAlloc)
		}
	}

	// deinit
	deinit {
		do {
			// select the device
			try device.select()

			// make sure pending queued commands complete before destroying the queue
			try blockCallerUntilComplete()

			// destroy the stream
			try cudaCheck(status: cudaStreamDestroy(handle))

			// remove from object tracking
			objectTracker.remove(trackingId: trackingId)
		} catch {
			writeLog(String(describing: error))
		}

		if willLog(level: .diagnostic) == true {
			diagnostic("\(releaseString) \(label)", categories: .streamAlloc)
		}
	}

	//----------------------------------------------------------------------------
	// properties
	public private(set) var trackingId = 0
	public let label: String
	public let device: ComputeDevice
	public let handle: cudaStream_t
	public let cudnn:  CudnnHandle
	public let cublas: CublasHandle
	public private(set) var id: Int

	// logging
	public var logLevel = LogLevel.error
	public var nestingLevel = 0
	public weak var currentLog: Log?

	//----------------------------------------------------------------------------
	// blockCallerUntilComplete
	public func blockCallerUntilComplete() throws {
		if willLog(level: .diagnostic) {
			diagnostic("\(blockString) \(label) blocking caller until complete",
				categories: .streamSync)
		}
		do {
			try cudaCheck(status: cudaStreamSynchronize(handle))
		} catch {
			writeLog("\(syncString) \(label) \(String(describing: error))")
			throw ModelError.evaluationFailed
		}
	}

	//----------------------------------------------------------------------------
	// createEvent
	public func createEvent(options: StreamEventOptions) throws -> StreamEvent {
		try device.select()
		return try CudaStreamEvent(options: options)
	}

	//----------------------------------------------------------------------------
	// delay the stream for event testing
	public func delay(seconds: Double) throws {
		let clockRate = (device as! CudaDevice).props.clockRate
		try cudaCheck(status: cudaDelayStream(seconds, clockRate, handle))
	}

	//----------------------------------------------------------------------------
	// record
	public func record(event: StreamEvent) throws -> StreamEvent {
		if willLog(level: .diagnostic) {
			diagnostic(
				"\(recordString) \(label) recording StreamEvent(\(event.trackingId))",
				categories: .streamSync)
		}
		try device.select()
		let event = event as! CudaStreamEvent
		try cudaCheck(status: cudaEventRecord(event.handle, handle))
		return event
	}

	//----------------------------------------------------------------------------
	// sync(with other
	public func sync(with other: DeviceStream, event: StreamEvent) throws {
		// only record an event if the streams are different
		guard id != other.id else { return }

		let otherStream = other as! CudaStream
		try device.select()
		let event = event as! CudaStreamEvent

		if willLog(level: .diagnostic) {
			diagnostic("\(syncString) \(label) synchronizing \(other.label)",
				categories: .streamSync)
		}

		// make sure the event object is clear in case it is being reused
		// so that we don't accidentally overwrite it
		try cudaCheck(status: cudaEventSynchronize(event.handle))

		// record the event on the other stream then wait
		try cudaCheck(status: cudaEventRecord(event.handle, otherStream.handle))
		try cudaCheck(status: cudaStreamWaitEvent(handle, event.handle, 0))
	}

	//----------------------------------------------------------------------------
	// wait(for event
	public func wait(for event: StreamEvent) throws {
		if willLog(level: .diagnostic) {
			diagnostic(
				"\(waitString) \(label) waiting for StreamEvent(\(event.trackingId))",
				categories: .streamSync)
		}
		try device.select()
		let event = event as! CudaStreamEvent
		try cudaCheck(status: cudaStreamWaitEvent(handle, event.handle, 0))
	}

	//----------------------------------------------------------------------------
	// device net functions
	public func createComputable(type: String, props: ComputableFilterProperties) throws -> Computable {
		try device.select()
		switch type {
		case "Activation"     : return CudaActivation(log: currentLog, props: props, stream: self)
		case "BatchNormalize" : return CudaBatchNormalize(log: currentLog, props: props, stream: self)
		case "Convolution"    : return CudaConvolution(log: currentLog, props: props, stream: self)
		case "Dropout"        : return CudaDropout(log: currentLog, props: props, stream: self)
		case "FullyConnected" : return CudaFullyConnected(log: currentLog, props: props, stream: self)
		case "LrnCrossChannel": return CudaLrnCrossChannel(log: currentLog, props: props, stream: self)
		case "Pooling"        : return CudaPooling(log: currentLog, props: props, stream: self)
		case "Softmax"        : return CudaSoftmax(log: currentLog, props: props, stream: self)

		default:
			props.writeLog("Unrecognized Computable class: \(type)")
			throw ModelError.setupFailed
		}
	}

	//----------------------------------------------------------------------------
	// createReductionContext
	public func createReductionContext(
		op: ReductionOp, dataType: DataType,
		inShape: Shape, outShape: Shape) throws -> ReductionContext {

		return try CudaReductionContext(
			stream: self, op: op, dataType: dataType,
			inTensor: TensorDescriptor(shape: inShape, dataType: dataType),
			outTensor: TensorDescriptor(shape: outShape, dataType: dataType))
	}

	//----------------------------------------------------------------------------
	// validate
	public func validate(data: DataView, hasRangeError result: inout DataView) throws {
		// validate
//		assert(data.rank == 1 || result.rank == 1)

		try cudaCheck(status: cudaValidateRange(
			cudaDataShape(from: data.flattened()), data.flattened().ro(using: self),
			cudaDataShape(from: result), result.rw(using: self),
			handle))
	}

	//----------------------------------------------------------------------------
	// asum
	//  Cublas always returns the result to host memory
	//  TODO: investigate value of device memory kernel
	// cublasSetPointerMode is a bad option
	public func asum(x: DataView, result: inout DataView) throws {
		// validate
		assert(x.rank == 1 && result.isScalar)
		assert(x.dataType == result.dataType)
		try device.select()

		switch x.dataType {
		case .real16F: fatalError("not implemented")
		case .real32F:
			try cudaCheck(status:	cublasSasum_v2(
				cublas.handle,
				Int32(x.items),
				x.roReal32F(using: self),
				Int32(x.itemStride),
				result.rwReal32F().baseAddress))
			
		case .real64F:
			try cudaCheck(status:	cublasDasum_v2(
				cublas.handle,
				Int32(x.items),
				x.roReal64F(using: self),
				Int32(x.itemStride),
				result.rwReal64F().baseAddress))
			
		default: fatalError()
		}
	}
	
	//----------------------------------------------------------------------------
	// compareEqual
	//  The result = a == b ? 1 : 0
	public func compareEqual(data aData: DataView, with bData: DataView,
	                         result: inout DataView) throws {
	  // for now just handle rank 1
	  assert(aData.rank == 1 && bData.rank == 1 && result.rank == 1)
	  assert(aData.items == bData.items && bData.items == result.items)
	  try device.select()
		try cudaCompareEqual(
			aData.elementCount,
			aData.dataType.cuda,
			aData.ro(using: self), aData.itemStride,
			bData.ro(using: self), bData.itemStride,
			result.dataType.cuda,
			result.rw(using: self), result.itemStride,
			handle
		)
  }
	
	//----------------------------------------------------------------------------
	// copy
	public func copy(from inData: DataView, to outData: inout DataView,
	                 normalizeInts: Bool = false) throws {
		// check for simple byte copy
		if !normalizeInts && inData.dataType == outData.dataType &&
			 inData.shape == outData.shape {

			// TODO: investigate performance of simply copying inData to outData
			// copying the DataView is a zero memcpy operation until the next time
			// inData is updated, which causes a mutation and copy anyway.
			// probably more expensive
			try cudaCheck(status: cudaMemcpyAsync(
				outData.rw(using: self),
				inData.ro(using: self),
				outData.viewByteCount,
				Cuda.cudaMemcpyDeviceToDevice, handle))
		} else {
			// TODO: write kernel for this
			try cpuCopy(from: inData, to: &outData, normalizeInts: normalizeInts)
		}
	}

	//----------------------------------------------------------------------------
	// reduce
	public func reduce(context: ReductionContext,
	                   inData: DataView,
	                   outData: inout DataView) throws {
		// cast to cuda
		let ctx = context as! CudaReductionContext

		try cudaCheck(status: cudnnReduceTensor(
			cudnn.handle,
			ctx.reduceTensorDesc,
			nil, 0,
			ctx.workspace.data,
			ctx.workspaceSizeInBytes,
			inData.one,
			ctx.inTensor.desc,
			inData.ro(using: self),
			outData.zero,
			ctx.outTensor.desc,
			outData.rw(using: self)
		))
	}

	//----------------------------------------------------------------------------
	// reduce
	//  includes indices for min/max operations
	public func reduce(context: ReductionContext,
	                   inData: DataView,
	                   outData: inout DataView,
	                   indices: inout DataView) throws {
		// cast to cuda
		let ctx = context as! CudaReductionContext

		// for now the indices must be Int32
		assert(indices.dataType == .real32I, "only Int32 indices are currently supported")
		assert(ctx.op == .min || ctx.op == .max, "indices may only be computed for min/max")

		try cudaCheck(status: cudnnReduceTensor(
			cudnn.handle,
			ctx.reduceTensorDesc,
			indices.rw(using: self),
			indices.viewByteCount,
			ctx.workspace.data,
			ctx.workspaceSizeInBytes,
			inData.one,
			ctx.inTensor.desc,
			inData.ro(using: self),
			outData.zero,
			ctx.outTensor.desc,
			outData.rw(using: self)
		))
	}

	//----------------------------------------------------------------------------
	// dot
	public func dot(x: DataView, y: DataView, result: inout DataView) throws {
		// validate
		assert(x.rank == 1 && y.rank == 1 && result.isScalar)
		assert(x.dataType == result.dataType && x.dataType == y.dataType)
		try device.select()

		switch x.dataType {
		case .real16F: fatalError("not implemented")
		case .real32F:
			try cudaCheck(status:	cublasSdot_v2(
				cublas.handle,
				Int32(x.items),
				x.roReal32F(using: self), Int32(x.itemStride),
				y.roReal32F(using: self), Int32(y.itemStride),
				result.rwReal32F().baseAddress))

		case .real64F:
			try cudaCheck(status:	cublasDdot_v2(
				cublas.handle,
				Int32(x.items),
				x.roReal64F(using: self), Int32(x.itemStride),
				y.roReal64F(using: self), Int32(y.itemStride),
				result.rwReal64F().baseAddress))

		default: fatalError()
		}
	}

	//----------------------------------------------------------------------------
	// softmaxLabelGradient
	public func softmaxLabelGradient(outData: DataView, labels: DataView,
	                                 inGrad: inout DataView) throws {
		try device.select()
		let N = outData.rows

		try cudaCheck(status: cudaSoftmaxLabelGradient(
			N,
			outData.dataType.cuda,
			outData.ro(using: self),
			outData.cols,
			outData.rowStride,
			labels.dataType.cuda,
			labels.ro(using: self),
			1.0 / Double(N),
			inGrad.dataType.cuda,
			inGrad.rw(using: self),
			handle))
	}

	//----------------------------------------------------------------------------
	// expand
	public func expand(labels: DataView, to expanded: inout DataView) throws {
		// validate
		assert(labels.dataType == expanded.dataType)
		assert(labels.rank == 1 && expanded.rows == labels.items)
		try device.select()

		try cudaCheck(status: cudaExpandLabels(
			expanded.dataType.cuda,
			labels.ro(using: self),
			labels.itemStride,
			expanded.rw(using: self),
			expanded.extent,
			expanded.strides,
			handle))
	}

	//----------------------------------------------------------------------------
	// update(weights
	public func update(weights: inout DataView, gradient: DataView,
	                   learningRate: Double) throws {
		// validate
		assert(
			weights.dataType == gradient.dataType &&
				weights.rank == 1 && gradient.rank == 1 &&
				weights.elementCount == gradient.elementCount)
		try device.select()

		try cudaCheck(status: cudaUpdateGradient(
			weights.dataType.cuda,
			weights.elementCount,
			weights.rw(using: self),
			gradient.ro(using: self),
			learningRate,
			handle))
	}

	//----------------------------------------------------------------------------
	// update(weights
	//	x[i] = y[i] = alpha * x[i] + beta * y[i]
	public func update(weights: inout DataView,
	                   gradient: DataView,
	                   learningRate: Double,
	                   history: inout DataView,
	                   momentum: Double) throws {
		// validate
		assert(
			weights.dataType == gradient.dataType &&
			weights.rank == 1 && gradient.rank == 1 &&
			weights.elementCount == gradient.elementCount &&
		  history.elementCount == gradient.elementCount)
		try device.select()

		try cudaCheck(status: cudaUpdateGradientWithMomentum(
			weights.dataType.cuda,
			weights.elementCount,
			weights.rw(using: self),
			gradient.ro(using: self),
			learningRate,
			history.rw(using: self),
			momentum,
			handle))
	}

//----------------------------------------------------------------------------
	// axpy
	//	y = alpha * x + y
	public func axpy(alpha: Double, x: DataView, y: inout DataView) throws {
		// validate
		assert(x.rank == 1 && y.rank == 1 && x.items == y.items)
		try device.select()

		let alpha = AnyValue(dataType: x.dataType, value: alpha)
		let count = Int32(x.items)
		let incx = Int32(x.itemStride)
		let incy = Int32(y.itemStride)
		let executionType = y.dataType == .real16F ?
			DataType.real32F.cuda : y.dataType.cuda

		try cudaCheck(status: cublasAxpyEx(
			cublas.handle, count,
			alpha.real32FPointer, alpha.dataType.cuda,
			x.ro(using: self), x.dataType.cuda, incx,
			y.rw(using: self), y.dataType.cuda, incy,
			executionType))
	}
	
	//----------------------------------------------------------------------------
	// createRandomGeneratorState
	public func createRandomGeneratorState(for dataView: DataView, seed: UInt?)
		throws -> RandomGeneratorState {
		// TODO: remove isContiguous requirement
		assert(dataView.shape.isContiguous)
		return try CudaPseudoRandomGeneratorState(log: currentLog, using: self,
			count: dataView.elementCount, seed: seed, name: label)
	}

	//-----------------------------------
	// fill
	public func fill(data: inout DataView, with constant: Double) throws {
		try device.select()

		if constant == 0 && data.shape.isContiguous	{
			let buffer = try data.rw(using: self)
			try cudaCheck(status:
				cudaMemsetAsync(buffer, Int32(0), data.viewByteCount, handle))
		} else {
			try cpuFill(data: &data, with: constant)
		}
	}

	//-----------------------------------
	// fillWithIndex
	public func fillWithIndex(data: inout DataView, startingAt: Int) throws {
		try device.select()
		try cpuFillWithIndex(data: &data, startingAt: startingAt)
	}

	//-----------------------------------
	// fillGaussian
	public func fillGaussian(data: inout DataView, mean: Double, std: Double,
	                         generatorState: RandomGeneratorState) throws {
		try cpuFillGaussian(data: &data, mean: mean, std: std)
//		// validate
//		assert(generatorState is CudaRandomGeneratorState)
//		try device.select()
//
//		try cudaCheck(status: cudaFillGaussian(
//			cudaDataShape(from: data.flattened()), data.rw(using: self), mean, std,
//			(generatorState as! CudaRandomGeneratorState).handle, handle))
	}

	//-----------------------------------
	// fillMSRA
	public func fillMSRA(data: inout DataView, varianceNorm: FillVarianceNorm,
	                     generatorState: RandomGeneratorState) throws {
		try cpuFillMSRA(data: &data, varianceNorm: varianceNorm)
//		// validate
//		assert(generatorState is CudaRandomGeneratorState)
//		try device.select()
//		let n = computeVarianceNorm(shape: data.shape, varianceNorm: varianceNorm)
//		try fillGaussian(data: &data, mean: 0, std: sqrt(2.0 / n),
//			               generatorState: generatorState)
	}

	//-----------------------------------
	// fillUniform
	public func fillUniform(data: inout DataView, range: ClosedRange<Double>,
	                        generatorState: RandomGeneratorState) throws {
		try cpuFillUniform(data: &data, range: range)
//		// validate
//		assert(generatorState is CudaRandomGeneratorState)
//		// TODO: remove this constraint from kernel
//		assert(data.shape.isContiguous)
//		try device.select()
//
//		try cudaCheck(status: cudaFillUniform(
//			cudaDataShape(from: data.flattened()), data.rw(using: self),
//			(generatorState as! CudaRandomGeneratorState).handle, handle))
	}

	//-----------------------------------
	// fillXavier
	public func fillXavier(data: inout DataView, varianceNorm: FillVarianceNorm,
	                       generatorState: RandomGeneratorState) throws {
		try cpuFillXavier(data: &data, varianceNorm: varianceNorm)
//		// validate
//		assert(generatorState is CudaRandomGeneratorState)
//		try device.select()
//		let n = computeVarianceNorm(shape: data.shape, varianceNorm: varianceNorm)
//
//		try cudaCheck(status: cudaFillXavier(
//			cudaDataShape(from: data.flattened()), data.rw(using: self), n,
//			(generatorState as! CudaRandomGeneratorState).handle, handle))
	}
	
	//============================================================================
	// gemm
	//    Row major matrix multiply.
	// A(m x k) * B(k x n) -> C(m x n)
	//
	// http://www.christophlassner.de/using-blas-from-c-with-row-major-data.html
	// dgemm("N","N", n, m, k, alpha, B, n, A, k, beta, C, n);
	public func gemm(alpha: Double = 1, transA: TransposeOp, matrixA: DataView,
	                 transB: TransposeOp, matrixB: DataView,
	                 beta: Double = 0, matrixC: inout DataView) throws {
		// make sure we are doing a 2D operation
		assert(matrixA.dataType == matrixB.dataType && matrixA.dataType == matrixC.dataType)
		try device.select()

		let alpha = AnyValue(dataType: matrixA.dataType, value: alpha)
		let beta  = AnyValue(dataType: matrixA.dataType, value: beta)

		let m = (transA == .noTranspose) ? matrixA.rows : matrixA.cols
		let k = (transA == .noTranspose) ? matrixA.cols : matrixA.rows
		let n = (transB == .noTranspose) ? matrixB.cols : matrixB.rows
		let rowStrideA = Int32(matrixA.rowStride)
		let rowStrideB = Int32(matrixB.rowStride)
		let rowStrideC = Int32(matrixC.rowStride)
		
		// TODO: there are no docs for this, read about cublasGemmAlgo_t
		switch matrixC.dataType {
		case .real16F:
			try cudaCheck(status:	cublasGemmEx(
				cublas.handle,
				transB.cublas, transA.cublas,
				Int32(n), Int32(m), Int32(k),
				alpha.real32FPointer,
				matrixB.roReal16F(using: self), matrixB.dataType.cuda, rowStrideB,
				matrixA.roReal16F(using: self), matrixA.dataType.cuda, rowStrideA,
				beta.real32FPointer,
				matrixC.rwReal16F(using: self), matrixC.dataType.cuda, rowStrideC,
				DataType.real32F.cuda, CUBLAS_GEMM_DFALT))

		case .real32F:
			try cudaCheck(status:	cublasGemmEx(
				cublas.handle,
				transB.cublas, transA.cublas,
				Int32(n), Int32(m), Int32(k),
				alpha.real32FPointer,
				matrixB.roReal32F(using: self), matrixB.dataType.cuda, rowStrideB,
				matrixA.roReal32F(using: self), matrixA.dataType.cuda, rowStrideA,
				beta.real32FPointer,
				matrixC.rwReal32F(using: self), matrixC.dataType.cuda, rowStrideC,
				matrixC.dataType.cuda, CUBLAS_GEMM_DFALT))

		case .real64F:
			try cudaCheck(status:	cublasGemmEx(
				cublas.handle,
				transB.cublas, transA.cublas,
				Int32(n), Int32(m), Int32(k),
				alpha.real32FPointer,
				matrixB.roReal64F(using: self), matrixB.dataType.cuda, rowStrideB,
				matrixA.roReal64F(using: self), matrixA.dataType.cuda, rowStrideA,
				beta.real32FPointer,
				matrixC.rwReal64F(using: self), matrixC.dataType.cuda, rowStrideC,
				matrixC.dataType.cuda, CUBLAS_GEMM_DFALT))

		default: fatalError("not implemented")
		}
		// TODO: there are no docs for this, read about cublasGemmAlgo_t
	}
} // CudaStream

//==============================================================================
// cudaDataShape(from:)
public func cudaDataShape(from data: DataView) -> cudaShape_t {
	var ptr = UnsafeMutablePointer<cudaShape_t>.allocate(capacity: 1)
	defer { ptr.deinitialize(count: 1); ptr.deallocate(capacity: 1) }

	cudaInitCudaShape(
		&ptr.pointee,
		data.dataType.cuda,
		data.shape.layout.cudnn,
		data.extent.count,
		data.extent,
		data.strides,
		data.shape.elementCount)

	return ptr.pointee
}

//==============================================================================
// CudaReductionContext
public final class CudaReductionContext : ReductionContext {

	// initializers
	public init(stream: CudaStream,
	            op: ReductionOp,
	            dataType: DataType,
	            inTensor: TensorDescriptor,
	            outTensor: TensorDescriptor) throws {

		self.op = op
		self.inTensor = inTensor
		self.outTensor = outTensor

		var temp: cudnnReduceTensorDescriptor_t?
		try cudaCheck(status: cudnnCreateReduceTensorDescriptor(&temp))
		reduceTensorDesc = temp!

		let indicesAction = (op == .min || op == .max) ?
			Cuda.CUDNN_REDUCE_TENSOR_FLATTENED_INDICES :
			Cuda.CUDNN_REDUCE_TENSOR_NO_INDICES

		// adjust intermediate data type if needed
		var reductionDataType: DataType
		switch dataType {
		case .real16F: reductionDataType = .real32F
		default: reductionDataType = dataType
		}

		try cudaCheck(status: cudnnSetReduceTensorDescriptor(
			reduceTensorDesc,
			op.cudnn,
			reductionDataType.cudnn,
			Cuda.CUDNN_PROPAGATE_NAN,
			indicesAction,
			Cuda.CUDNN_32BIT_INDICES
		))

		// determine workspace size
		var tempWorkspaceSizeInBytes = 0
		try cudaCheck(status: cudnnGetReductionWorkspaceSize(
			stream.cudnn.handle,
			reduceTensorDesc,
			inTensor.desc,
			outTensor.desc,
			&tempWorkspaceSizeInBytes
		))
		workspaceSizeInBytes = tempWorkspaceSizeInBytes
		workspace = try stream.device.createArray(count: workspaceSizeInBytes)
	}

	//----------------------------------------------------------------------------
	// properties
	public let op: ReductionOp
	public let workspace: DeviceArray
	public let workspaceSizeInBytes: Int
	public let reduceTensorDesc: cudnnReduceTensorDescriptor_t
	public let inTensor: TensorDescriptor
	public let outTensor: TensorDescriptor
}
