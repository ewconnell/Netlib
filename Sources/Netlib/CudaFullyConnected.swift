//******************************************************************************
//  Created by Edward Connell on 8/24/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
import Cuda

public class CudaFullyConnected : Computable {
	// initializers
	public init(log: Log?, props: ComputableFilterProperties, stream: CudaStream) {
		self.props = (props as? FullyConnectedProperties)!
		currentLog = log
		dataStream = stream
		trackingId = objectTracker.register(type: self)
	}

	deinit { objectTracker.remove(trackingId: trackingId) }

	//----------------------------------------------------------------------------
	// properties
	public private(set) var trackingId = 0
	private weak var props: FullyConnectedProperties!
	public  var stream: DeviceStream { return dataStream }
	public  let dataStream: CudaStream
	private var biasTensor: TensorDescriptor!
	private var outTensor : TensorDescriptor!
	private var backReductionContext: ReductionContext!
	private var inputGradient: DataView!

	// logging
	public var logLevel = LogLevel.error
	public var nestingLevel = 0
	public weak var currentLog: Log?

	//----------------------------------------------------------------------------
	// forward
	public func forward(mode: EvaluationMode, inData: DataView, labels: DataView?,
	                    outData: inout DataView, backData: inout DataView?) throws {
		// flatten the input into a set of vectors
		try dataStream.gemm(
			transA: .noTranspose,	matrixA: inData.flattened(axis: 1),
			transB: .noTranspose, matrixB: props.weights.data,
			matrixC: &outData)
		
		try cudaCheck(status: cudnnAddTensor(
			dataStream.cudnn.handle,
			props.bias.data.one,
			biasTensor.desc,
			props.bias.data.ro(using: dataStream),
			outData.one,
			outTensor.desc,
			outData.rw(using: dataStream)))
	}
	
	//----------------------------------------------------------------------------
	// backward
	//  This computable uses two auxiliary streams for computing the weights
	// and bias updates. Each computable item needs to sequentially access
	// the shared weights and bias buffers.
	public func backward(outData: DataView, outGrad: DataView?,
	                     inData: DataView, inGrad: inout DataView?,
	                     solver: ModelSolver, labels: DataView?) throws {
		// data
		// inGrad is nil for inputs that don't do a backward (e.g. leaf elements)
		if var gradient = inGrad {
			// create once
			if inputGradient == nil {
				inputGradient = try gradient.referenceFlattened(axis: 1, using: dataStream)
			}
			try dataStream.gemm(transA: .noTranspose, matrixA: outGrad!,
				                  transB: .transpose, matrixB: props.weights.data,
				                  matrixC: &inputGradient!)
		}
	}

	//----------------------------------------------------------------------------
	// setupForward
	public func setupForward(mode: EvaluationMode, inData: DataView, labels: DataView?,
	                         outData: inout DataView, backData: inout DataView?) throws {
		// validate
		guard let outputChannels = props.outputChannels else {
			props.writeLog("outputChannels must be specified for \(props.namePath)")
			throw ModelError.setupFailed
		}

		// flatten input
		let flatInData = inData.flattened(axis: 1)

		// set weights shape
		try props.weights.setExtent([flatInData.cols, outputChannels], using: dataStream)

		// set bias shape
		try props.bias.setExtent([outputChannels], using: dataStream)
		let biasTensorShape = Shape(extent: [1, outputChannels, 1, 1], layout: .nchw)
		biasTensor = try props.bias.data.createTensorDescriptor(asShape: biasTensorShape)

		// set output
		outData = DataView(rows: flatInData.rows,
			                 cols: outputChannels,
			                 dataType: props.outDataType)

		let outTensorShape = Shape(extent: [outData.rows, outData.cols, 1, 1], layout: .nchw)
		outTensor = try outData.createTensorDescriptor(asShape: outTensorShape)
	}
	
	//----------------------------------------------------------------------------
	// setupBackward
	public func setupBackward(outData: DataView, outGrad: DataView?, inData: DataView) throws {
		// dataType
		let dataType = props.weights.data.dataType
		inputGradient = nil

		// weights
		let weightsShape = props.weights.data.shape
		props.weights.grad = DataView(shape: weightsShape, dataType: dataType)
		props.weights.grad.name = "\(props.namespaceName).weights.grad"

		// set weights gradient update function
		props.weights.setGradientUpdateFunction(using: dataStream) { [unowned self] _ in
			try self.dataStream.gemm(
				alpha: 1,
				transA: .transpose, matrixA: inData.flattened(axis: 1),
				transB: .noTranspose, matrixB: outGrad!,
				beta: 0,
				matrixC: &self.props.weights.grad)
		}

		// bias
		let biasShape = props.bias.data.shape
		props.bias.grad = DataView(shape: biasShape, dataType: dataType)
		props.bias.grad.name = "\(props.namespaceName).bias.grad"

		// setup back reduction context
		let biasDiffTensorShape = Shape(extent: [1, props.bias.grad.items, 1, 1], layout: .nchw)
		backReductionContext = try CudaReductionContext(
			stream: dataStream,
			op: .add,
			dataType: dataType,
			inTensor: outTensor,
			outTensor: props.bias.grad.createTensorDescriptor(asShape: biasDiffTensorShape))

		// bias gradient update function
		props.bias.setGradientUpdateFunction(using: dataStream) { [unowned self] _ in
			try self.dataStream.reduce(context: self.backReductionContext,
				                         inData: outGrad!, outData: &self.props.bias.grad)
		}
	}
} // CudaFullyConnected

