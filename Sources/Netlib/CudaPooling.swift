//******************************************************************************
//  Created by Edward Connell on 8/24/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
import Cuda

public class CudaPooling : Computable {
	// initializers
	public init(log: Log?, props: ComputableFilterProperties, stream: CudaStream) {
		self.props = props as! PoolingProperties
		currentLog = log
		dataStream = stream
		trackingId = objectTracker.register(type: self)
	}
	deinit { objectTracker.remove(trackingId: trackingId) }

	//----------------------------------------------------------------------------
	// properties
	public private(set) var trackingId = 0
	public var stream: DeviceStream { return dataStream }
	private weak var props: PoolingProperties!
	private var poolingDescriptor: PoolingDescriptor!
	private let dataStream: CudaStream
	private var inTensor: TensorDescriptor!
	private var outTensor: TensorDescriptor!

	// logging
	public var logLevel = LogLevel.error
	public var nestingLevel = 0
	public weak var currentLog: Log?

	//----------------------------------------------------------------------------
	// forward
	public func forward(mode: EvaluationMode, inData: DataView, labels: DataView?,
	                    outData: inout DataView, backData: inout DataView?) throws {

		try cudaCheck(status: cudnnPoolingForward(
			dataStream.cudnn.handle,
			poolingDescriptor.desc,
			inData.one,
			inTensor.desc,
			inData.ro(using: dataStream),
			outData.zero,
			outTensor.desc,
			outData.rw(using: dataStream)))
	}

	//----------------------------------------------------------------------------
	// backward
	public func backward(outData: DataView, outGrad: DataView?,
	                     inData: DataView, inGrad: inout DataView?,
	                     solver: ModelSolver, labels: DataView?) throws {
		// inGrad is nil for inputs that don't perform backward
		if inGrad == nil { return }

		try cudaCheck(status: cudnnPoolingBackward(
			dataStream.cudnn.handle,
			poolingDescriptor.desc,
			outData.one,
			outTensor.desc,
			outData.ro(using: dataStream),
			outTensor.desc,
			outGrad!.ro(using: dataStream),
			inTensor.desc,
			inData.ro(using: dataStream),
			inGrad!.zero,
			inTensor.desc,
			inGrad!.rw(using: dataStream)))
	}
	
	//----------------------------------------------------------------------------
	// setupForward
	public func setupForward(mode: EvaluationMode, inData: DataView, labels: DataView?,
	                         outData: inout DataView, backData: inout DataView?) throws {
		// create descriptor
		let poolingRank = inData.extent.count - 2
		let padding     = expand(array: props.pad, to: poolingRank)
		let windowSize  = expand(array: props.windowSize, to: poolingRank)
		let stride      = expand(array: props.stride, to: poolingRank)
		
		poolingDescriptor = try PoolingDescriptor(
			mode: props.mode, nan: props.nan, rank: poolingRank,
			window: windowSize, padding: padding, stride: stride)

		// create input tensor descriptor
		inTensor  = try inData.createTensorDescriptor()

		// assure the output is the correct type and size
		var outDims = [Int32](repeating: 0, count: inData.extent.count)
		try cudaCheck(status: cudnnGetPoolingNdForwardOutputDim(
			poolingDescriptor.desc, inTensor.desc, Int32(inData.extent.count), &outDims))

		// create output
		let outShape = Shape(extent: outDims.map { Int($0)})
		outData = DataView(shape: outShape, dataType: props.outDataType)
		outTensor = try outData.createTensorDescriptor()
	}
}


//==============================================================================
// Enum --> Cuda value mapping
//
extension PoolingMode {
	public var cudnn: cudnnPoolingMode_t {
		get {
			switch self {
			case .averageExcludePadding: return CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING
			case .averageIncludePadding: return CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING
			case .max                  : return CUDNN_POOLING_MAX
			}
		}
	}
}


