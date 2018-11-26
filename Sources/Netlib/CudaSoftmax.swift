//******************************************************************************
//  Created by Edward Connell on 8/24/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
import Cuda

public class CudaSoftmax : Computable {
	// initializers
	public init(log: Log?, props: ComputableFilterProperties, stream: CudaStream) {
		self.props = (props as? SoftmaxProperties)!
		currentLog = log
		dataStream = stream
		trackingId = objectTracker.register(type: self)
	}

	deinit {
		objectTracker.remove(trackingId: trackingId)
	}

	//----------------------------------------------------------------------------
	// properties
	public private(set) var trackingId = 0
	public var stream: DeviceStream { return dataStream }
	private weak var props: SoftmaxProperties!
	private let dataStream: CudaStream
	private var inTensor: TensorDescriptor!
	private var outTensor: TensorDescriptor!
	private var maxReductionContext: ReductionContext!
	private var maxValuesData = DataView()

	// logging
	public var logLevel = LogLevel.error
	public var nestingLevel = 0
	public weak var currentLog: Log?

	//----------------------------------------------------------------------------
	// setupForward
	public func setupForward(mode: EvaluationMode, inData: DataView, labels: DataView?,
	                         outData: inout DataView, backData: inout DataView?) throws {
		// create tensor descriptors for computing probabilities
		let tensorShape = inData.layout != .matrix ? inData.shape :
			Shape(extent: [inData.rows, inData.cols, 1, 1], layout: .nchw)

		inTensor = try inData.createTensorDescriptor(asShape: tensorShape)
		outTensor = inTensor

		// assure the output is the correct type and size
		if props.outputType == .probabilities && labels == nil {
			outData = DataView(shape: inData.shape, dataType: inData.dataType)

		} else {
			outData = DataView(count: inData.items,
				                 dataType: labels?.dataType ?? inData.dataType)
			backData = DataView(shape: inData.shape, dataType: inData.dataType)
		}
		// if there are labels then make sure the output extent matches
		assert(labels == nil || labels!.extent == outData.extent)
	}

	//----------------------------------------------------------------------------
	// forward
	//  Labels present
	//    forward stores the actual prediction in backData
	//  to be used during a backward learning pass. The output mode specifies
	//  whether labels or expanded label values are passed forward as the result.
	//
	//  Labels not present
	//    forward writes output to outData and is equivalent to calling predict
	//
	public func forward(mode: EvaluationMode, inData: DataView, labels: DataView?,
	                    outData: inout DataView, backData: inout DataView?) throws {

		if mode == .training {
			let outPointer = (props.outputType == .probabilities && labels == nil) ?
				try outData.rw(using: dataStream) :
				try backData!.rw(using: dataStream)

			try cudaCheck(status: cudnnSoftmaxForward(
				dataStream.cudnn.handle,
				props.algorithm.cudnn,
				props.mode.cudnn,
				inData.one,
				inTensor.desc,
				inData.ro(using: dataStream),
				outData.zero,
				outTensor.desc,
				outPointer))

			if props.outputType == .labels {
				if let labels = labels {
					try dataStream.copy(from: labels, to: &outData)
				} else {
					// find labels
					try getMaxIndices(of: backData!, indices: &outData)
				}
			} else if let labels = labels {
				// synthesize probabilities from ground truth labels
				try dataStream.expand(labels: labels, to: &outData)
			}

		} else {
			// if labels are desired then write to the backData buffer as
			// a temp workspace for finding the maxElement
			let outPointer = (props.outputType == .probabilities) ?
				try outData.rw(using: dataStream) :
				try backData!.rw(using: dataStream)

			try cudaCheck(status: cudnnSoftmaxForward(
				dataStream.cudnn.handle,
				props.algorithm.cudnn,
				props.mode.cudnn,
				inData.one,
				inTensor.desc,
				inData.ro(using: dataStream),
				outData.zero,
				outTensor.desc,
				outPointer))

			// find the labels
			if props.outputType == .labels {
				try getMaxIndices(of: backData!, indices: &outData)
			}
		}
	}

	//----------------------------------------------------------------------------
	// backward
	public func backward(outData: DataView, outGrad: DataView?,
	                     inData: DataView, inGrad: inout DataView?,
	                     solver: ModelSolver, labels: DataView?) throws {
		// inGrad is nil for inputs that don't perform backward
		if inGrad == nil { return }

		// if there are labels then use them to compute the input grad
		if let labels = labels {
			try dataStream.softmaxLabelGradient(outData: outData, labels: labels,
				                                  inGrad: &inGrad!)
		} else {
			// if there aren't any labels then do a normal backward
			try cudaCheck(status: cudnnSoftmaxBackward(
				dataStream.cudnn.handle,
				props.algorithm.cudnn,
				props.mode.cudnn,
				outData.one,
				outTensor.desc,
				outData.ro(using: dataStream),
				outTensor.desc,
				outGrad!.ro(using: dataStream),
				inGrad!.zero,
				inTensor.desc,
				inGrad!.rw(using: dataStream)))
		}
	}

	//----------------------------------------------------------------------------
	// getMaxIndices
	private func getMaxIndices(of data: DataView, indices: inout DataView) throws {
		// lazy create buffer and reduction context
		if maxReductionContext == nil {
			let dataTensorShape = Shape(extent: [data.rows, data.cols, 1, 1], layout: .nchw)
			let indicesTensorShape = Shape(extent: [indices.rows, 1, 1, 1], layout: .nchw)
			maxValuesData = DataView(shape: indicesTensorShape, dataType: data.dataType)

			maxReductionContext = try CudaReductionContext(
				stream: dataStream,
				op: .max,
				dataType: data.dataType,
				inTensor: data.createTensorDescriptor(asShape: dataTensorShape),
				outTensor: maxValuesData.createTensorDescriptor())
		}

		try dataStream.reduce(context: maxReductionContext,
			                    inData: data,
			                    outData: &maxValuesData,
			                    indices: &indices)
	}
}

//==============================================================================
// Enum --> Cuda value mapping
//
extension SoftmaxAlgorithm {
	public var cudnn: cudnnSoftmaxAlgorithm_t {
		get {
			switch self {
			case .accurate: return CUDNN_SOFTMAX_ACCURATE
			case .fast    : return CUDNN_SOFTMAX_FAST
			case .log     : return CUDNN_SOFTMAX_LOG
			}
		}
	}
}

extension SoftmaxMode {
	public var cudnn: cudnnSoftmaxMode_t {
		get {
			switch self {
			case .channel : return CUDNN_SOFTMAX_MODE_CHANNEL
			case .instance: return CUDNN_SOFTMAX_MODE_INSTANCE
			}
		}
	}
}


