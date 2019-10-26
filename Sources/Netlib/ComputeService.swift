//******************************************************************************
// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

//==============================================================================
// ComputeService
public protocol ComputeService : ObjectTracking, Logging {
	init(log: Log?) throws
	var devices: [ComputeDevice] { get }
	var id: Int { get set }
	var name: String { get }
}

//==============================================================================
// ComputeDevice
//    This specifies the compute device interface
public protocol ComputeDevice : ObjectTracking, Logging {
	var attributes: [String:String] { get }
	var availableMemory: Int { get }
	var maxThreadsPerBlock: Int { get }
	var name: String { get }
	var id: Int { get }
	var service: ComputeService! { get }
	var usesUnifiedAddressing: Bool { get }

	//----------------------------------------------------------------------------
	// device resource functions
	func createArray(count: Int) throws -> DeviceArray
	func createStream(label: String) throws -> DeviceStream
	func select() throws
	func supports(dataType: DataType) -> Bool
}

public enum ComputeError : Error {
	case serviceIsUnavailable
	case functionFailure(location: String, message: String)
}

//==============================================================================
// AnyValue
//  This is used to provide the necessary pointers to scale factors
// needed by cuda. Kind of ugly...
public final class AnyValue {
	public init(dataType suggestedDataType: DataType, value: Double) {
		self.dataType = suggestedDataType == .real64F ? .real64F : .real32F
		
		switch dataType {
		case .real32F:
			floatValue = Float(value)
			valuePointer = UnsafeRawPointer(withUnsafePointer(to: &floatValue) { $0 })
			
		case .real64F:
			doubleValue = value
			valuePointer = UnsafeRawPointer(withUnsafePointer(to: &doubleValue) { $0 })
			
		default: fatalError()
		}
	}
	
	public var pointer : UnsafeRawPointer { return valuePointer }
	
	public var real32FPointer : UnsafePointer<Float> {
		assert(dataType == .real32F)
		return withUnsafePointer(to: &floatValue) { $0 }
	}
	
	public var real64FPointer : UnsafePointer<Double> {
		assert(dataType == .real64F)
		return withUnsafePointer(to: &doubleValue) { $0 }
	}
	
	private let valuePointer: UnsafeRawPointer
	public  let dataType: DataType
	private var floatValue: Float = 0
	private var doubleValue: Double = 0
}

//==============================================================================
// Computable
public protocol Computable : ObjectTracking, Logging {
	var stream: DeviceStream { get }

	func setupForward(mode: EvaluationMode, inData: DataView, labels: DataView?,
	                  outData: inout DataView, backData: inout DataView?) throws

	func forward(mode: EvaluationMode, inData: DataView, labels: DataView?,
	             outData: inout DataView, backData: inout DataView?) throws

	func setupBackward(outData: DataView, outGrad: DataView?, inData: DataView) throws

	func backward(outData: DataView, outGrad: DataView?,
	              inData: DataView, inGrad: inout DataView?,
	              solver: ModelSolver, labels: DataView?) throws
}

extension Computable {
	public func setupBackward(outData: DataView, outGrad: DataView?, inData: DataView) throws {}
}

//==============================================================================
// DeviceArray
//    This represents a device data array
public protocol DeviceArray : ObjectTracking, Logging {
	var device: ComputeDevice { get }
	var data: UnsafeMutableRawPointer { get }
	var count: Int { get }
	var version: Int { get set }

	func zero(using stream: DeviceStream?) throws
	func copyAsync(from other: DeviceArray, using stream: DeviceStream) throws
	func copyAsync(from buffer: BufferUInt8, using stream: DeviceStream) throws
	func copy(to buffer: MutableBufferUInt8, using stream: DeviceStream) throws
	func copyAsync(to buffer: MutableBufferUInt8, using stream: DeviceStream) throws
}

//==============================================================================
// StreamEvent
public protocol StreamEvent : ObjectTracking {
	init(options: StreamEventOptions) throws
	var occurred: Bool { get }
}

public struct StreamEventOptions: OptionSet {
	public init(rawValue: Int) { self.rawValue = rawValue }
	public let rawValue: Int
	public static let hostSync     = StreamEventOptions(rawValue: 1 << 0)
	public static let timing       = StreamEventOptions(rawValue: 1 << 1)
	public static let interprocess = StreamEventOptions(rawValue: 1 << 2)
}

//==============================================================================
// RandomGeneratorState
public protocol RandomGeneratorState : ObjectTracking {
	var count: Int { get }
}

//==============================================================================
// ComputableFilterProperties
public protocol ComputableFilterProperties : Filter { }

//==============================================================================
// DeviceStream
//	A device stream is an asynchronous queue of commands executed on
// the associated device
//
public protocol DeviceStream : ObjectTracking, Logging {
	// properties
	var device: ComputeDevice { get }
	var label: String { get }
	var id: Int { get }

	//-------------------------------------
	// synchronization
	func blockCallerUntilComplete() throws
	func createEvent(options: StreamEventOptions) throws -> StreamEvent
	func delay(seconds: Double) throws
	func record(event: StreamEvent) throws -> StreamEvent
	func sync(with other: DeviceStream, event: StreamEvent) throws
	func wait(for event: StreamEvent) throws

	//-------------------------------------
	// device optimized computabes
	func createComputable(type: String, props: ComputableFilterProperties) throws -> Computable

	//-------------------------------------
	// reduction state
	func createReductionContext(op: ReductionOp, dataType: DataType,
	                            inShape: Shape, outShape: Shape) throws -> ReductionContext

	//-------------------------------------
	// validate
	//  TODO: right now this tests for all values being finite or not
	//  eventually add range parameter
	func validate(data: DataView, hasRangeError: inout DataView) throws

	//-------------------------------------
	// functions
	func asum(x: DataView, result: inout DataView) throws

	func compareEqual(data aData: DataView, with bData: DataView, result: inout DataView) throws

	func copy(from inData: DataView, to outData: inout DataView, normalizeInts: Bool) throws

	func reduce(context: ReductionContext, inData: DataView,
	            outData: inout DataView) throws

	func reduce(context: ReductionContext, inData: DataView,
	            outData: inout DataView, indices: inout DataView) throws

	func dot(x: DataView, y: DataView, result: inout DataView) throws

	func expand(labels: DataView, to expanded: inout DataView) throws

	// update
	func update(weights: inout DataView, gradient: DataView,
	            learningRate: Double) throws

	// with momentum
	func update(weights: inout DataView, gradient: DataView, learningRate: Double,
	            history: inout DataView, momentum: Double) throws

	// alpha * x + y
	func axpy(alpha: Double, x: DataView, y: inout DataView) throws

	// random number generator state
	func createRandomGeneratorState(for dataView: DataView, seed: UInt?) throws -> RandomGeneratorState

	// fill
	func fill(data: inout DataView, with constant: Double) throws
	func fillWithIndex(data: inout DataView, startingAt: Int) throws
	func fillGaussian(data: inout DataView, mean: Double, std: Double, generatorState: RandomGeneratorState) throws
	func fillMSRA(data: inout DataView, varianceNorm: FillVarianceNorm, generatorState: RandomGeneratorState) throws
	func fillUniform(data: inout DataView, range: ClosedRange<Double>, generatorState: RandomGeneratorState) throws
	func fillXavier(data: inout DataView, varianceNorm: FillVarianceNorm, generatorState: RandomGeneratorState) throws

	// A x B -> C
	func gemm(alpha: Double, transA: TransposeOp, matrixA: DataView,
	          transB: TransposeOp, matrixB: DataView,
	          beta: Double, matrixC: inout DataView) throws
}

public enum FillVarianceNorm : String, EnumerableType { case fanIn, fanOut, average }
public enum TransposeOp { case transpose, noTranspose, conjugateTranspose }
public enum ReductionOp { case add, mul, min, max, amax, avg, norm1, norm2 }
public protocol ReductionContext {}

extension DeviceStream {
	func fillGaussian(data: inout DataView, mean: Double, std: Double, seed: UInt? = nil) throws {
		try fillGaussian(data: &data, mean: mean, std: std,
			generatorState: createRandomGeneratorState(for: data, seed: seed))
	}

	func fillMSRA(data: inout DataView, varianceNorm: FillVarianceNorm, seed: UInt? = nil) throws {
		try fillMSRA(data: &data, varianceNorm: varianceNorm,
			generatorState: createRandomGeneratorState(for: data, seed: seed))
	}

	func fillUniform(data: inout DataView, range: ClosedRange<Double>, seed: UInt? = nil) throws {
		try fillUniform(data: &data, range: range,
			generatorState: createRandomGeneratorState(for: data, seed: seed))
	}

	func fillXavier(data: inout DataView, varianceNorm: FillVarianceNorm, seed: UInt? = nil) throws {
		try fillXavier(data: &data, varianceNorm: varianceNorm,
			generatorState: createRandomGeneratorState(for: data, seed: seed))
	}
}
