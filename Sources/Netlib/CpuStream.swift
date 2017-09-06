//******************************************************************************
//  Created by Edward Connell on 4/5/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
public final class CpuStream : DeviceStream {
	// initializers
	public init(log: Log?, device: ComputeDevice, id: Int, label: String) throws {
		currentLog  = log
		self.device = device
		self.id     = id
		self.label  = label
		trackingId  = objectTracker.register(type: self)
	}
	deinit { objectTracker.remove(trackingId: trackingId) }

	//----------------------------------------------------------------------------
	// properties
	public private(set) var trackingId = 0
	public let device: ComputeDevice
	public let label: String
	public let id: Int

	// logging
	public var logLevel = LogLevel.error
	public var nestingLevel = 0
	public weak var currentLog: Log?

	//----------------------------------------------------------------------------
	// createEvent
	public func createEvent(options: StreamEventOptions) throws -> StreamEvent {
		return CpuStreamEvent(options: options)
	}

	public func delay(seconds: Double) throws {
	}

	// blockCallerUntilComplete
	public func blockCallerUntilComplete() throws {
		
	}
	
	public func wait(for event: StreamEvent) throws {
		
	}

	// sync(with other
	public func sync(with other: DeviceStream, event: StreamEvent) throws {
	}

	public func record(event: StreamEvent) throws  -> StreamEvent {
		return event
	}

	// select
	public func select() throws {
		try device.select()
	}

	//----------------------------------------------------------------------------
	// device net functions
	public func createComputable(type: String, props: ComputableFilterProperties) throws -> Computable
	{
		return FunctionStub(log: currentLog, stream: self)
	}

	//----------------------------------------------------------------------------
	// createReductionContext
	public func createReductionContext(op: ReductionOp,
	                                   dataType: DataType,
	                                   inShape: Shape,
	                                   outShape: Shape) throws -> ReductionContext {

		fatalError()
	}

	//----------------------------------------------------------------------------
	// simple math functions
	public func asum(x: DataView, result: inout DataView) throws {
		try cpuAsum(x: x, result: &result)
	}

	public func compareEqual(data aData: DataView, with bData: DataView,
	                         result: inout DataView) throws
	{

	}

	public func copy(from inData: DataView, to outData: inout DataView,
	                      normalizeInts: Bool) throws {
		try cpuCopy(from: inData, to: &outData, normalizeInts: normalizeInts)
	}

	//----------------------------------------------------------------------------
	// reduce
	public func reduce(context: ReductionContext,
	                   inData: DataView, outData: inout DataView) throws {
	}

	public func reduce(context: ReductionContext,
	                   inData: DataView,
	                   outData: inout DataView,
	                   indices: inout DataView) throws {
	}

	public func dot(x: DataView, y: DataView, result: inout DataView) throws {

	}


	public func expand(labels: DataView, to expanded: inout DataView) throws {
		try cpuExpand(labels: labels, to: &expanded)
	}

	//----------------------------------------------------------------------------
	// update
	public func update(weights: inout DataView, gradient: DataView,
	                   learningRate: Double) throws {
		assert(gradient.dataType == weights.dataType)

	}

	public func update(weights: inout DataView, gradient: DataView, learningRate: Double,
	            history: inout DataView, momentum: Double) throws {

	}

	// alpha * x + y
	public func axpy(alpha: Double, x: DataView, y: inout DataView) throws {
		try cpuAxpy(alpha: alpha, x: x, y: &y)
	}
	
	public func validate(data: DataView, hasRangeError: inout DataView) throws {

	}

	//----------------------------------------------------------------------------
	// createRandomGeneratorState
	public func createRandomGeneratorState(for dataView: DataView, seed: UInt?) throws
			-> RandomGeneratorState {
		return CpuRandomGeneratorState()
	}

	//----------------------------------------------------------------------------
	// fill
	public func fill(data: inout DataView, with constant: Double) throws {
		try cpuFill(data: &data, with: constant)
	}

	public func fillWithIndex(data: inout DataView, startingAt: Int) throws {
		try cpuFillWithIndex(data: &data, startingAt: startingAt)
	}

	public func fillGaussian(data: inout DataView, mean: Double, std: Double,
	                         generatorState: RandomGeneratorState) throws {
		try cpuFillGaussian(data: &data, mean: mean, std: std)
	}
	
	public func fillMSRA(data: inout DataView, varianceNorm: FillVarianceNorm,
	                     generatorState: RandomGeneratorState) throws {
		try cpuFillMSRA(data: &data, varianceNorm: varianceNorm)
	}
	
	public func fillUniform(data: inout DataView, range: ClosedRange<Double>,
	                        generatorState: RandomGeneratorState) throws {
		try cpuFillUniform(data: &data, range: range)
	}
	
	public func fillXavier(data: inout DataView, varianceNorm: FillVarianceNorm,
	                       generatorState: RandomGeneratorState) throws {
		try cpuFillXavier(data: &data, varianceNorm: varianceNorm)
	}
	
	//----------------------------------------------------------------------------
	// gemm
	// A x B -> C
	public func gemm(alpha: Double, transA: TransposeOp, matrixA: DataView,
	                 transB: TransposeOp, matrixB: DataView,
	                 beta: Double, matrixC: inout DataView) throws
	{
		cpuGemm(alpha: alpha, transA: transA, matrixA: matrixA,
		        transB: transB, matrixB: matrixB, beta: beta, matrixC: &matrixC)
	}
}

//==============================================================================
// CpuStreamEvent
final public class CpuStreamEvent : StreamEvent {
	public required init(options: StreamEventOptions) {
		trackingId = objectTracker.register(type: self)
	}
	deinit { objectTracker.remove(trackingId: trackingId) }

	//----------------------------------------------------------------------------
	// properties
	public private (set) var trackingId = 0
	public var occurred: Bool { return true }
}


//==============================================================================
// FunctionStub
public class FunctionStub : Computable {
	// initializers
	public init(log: Log?, stream: CpuStream) {
		currentLog = log
		dataStream = stream
		trackingId = objectTracker.register(type: self)
	}
	deinit { objectTracker.remove(trackingId: trackingId) }

	//----------------------------------------------------------------------------
	// properties
	public private(set) var trackingId = 0
	public var stream: DeviceStream { return dataStream }
	let dataStream: CpuStream

	// logging
	public var logLevel = LogLevel.error
	public var nestingLevel = 0
	public weak var currentLog: Log?

	//----------------------------------------------------------------------------
	// properties
	public func setupForward(mode: EvaluationMode, inData: DataView, labels: DataView?,
	                         outData: inout DataView, backData: inout DataView?) throws {
		outData = DataView(shape: inData.shape, dataType: inData.dataType)
	}
	public func setupBackward(outData: DataView, outGrad: DataView?, inData: DataView) throws {	}
	
	public func forward(mode: EvaluationMode, inData: DataView, labels: DataView?,
	                    outData: inout DataView, backData: inout DataView?) throws {
	}
	public func backward(outData: DataView, outGrad: DataView?, inData: DataView,
	                     inGrad: inout DataView?, solver: ModelSolver, labels: DataView?) throws {}
	
}

//==============================================================================
// CpuRandomGeneratorState
public class CpuRandomGeneratorState : RandomGeneratorState {
	public var count = 0
	public private(set) var trackingId = 0
}
