//******************************************************************************
//  Created by Edward Connell on 8/24/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
import Cuda

public final class CudaBatchNormalize : Computable {
	// initializers
	public init(log: Log?, props: ComputableFilterProperties, stream: CudaStream) {
		self.props = props as! BatchNormalizeProperties
		currentLog = log
		dataStream = stream
		trackingId = objectTracker.register(type: self)
	}

	deinit { objectTracker.remove(trackingId: trackingId) }
	
	//----------------------------------------------------------------------------
	// properties
	public private(set) var trackingId = 0
	public var stream: DeviceStream { return dataStream }
	private let dataStream: CudaStream
	private weak var props: BatchNormalizeProperties!
	private var inTensor: TensorDescriptor!
	private var outTensor: TensorDescriptor!
	private var scaleBiasMeanVarianceTensor: TensorDescriptor!
	private var bnMode: BatchNormalizeMode!
	private var workspaceSize = 0

	// working buffers
	private var running_mean: DeviceArray!
	private var running_var: DeviceArray!
	private var saved_mean: DeviceArray!
	private var saved_var: DeviceArray!
	private var grad_scale: DeviceArray!
	private var grad_bias: DeviceArray!
	private var scale: DeviceArray!
	private var bias: DeviceArray!

	// logging
	public var logLevel = LogLevel.error
	public var nestingLevel = 0
	public weak var currentLog: Log?

	//----------------------------------------------------------------------------
	// forward
	public func forward(mode: EvaluationMode, inData: DataView, labels: DataView?,
	                    outData: inout DataView, backData: inout DataView?) throws {
		if mode == .training {
			let expAverageFactor = 1.0 - props.momentum

			try cudaCheck(status: cudnnBatchNormalizationForwardTraining(
				dataStream.cudnn.handle,
				bnMode.cudnn,
				inData.one,
				outData.zero,
				inTensor.desc,
				inData.ro(using: dataStream),
				outTensor.desc,
				outData.rw(using: dataStream),
				scaleBiasMeanVarianceTensor!.desc,
				scale.data,
				bias.data,
				expAverageFactor,
				running_mean.data,
				running_var.data,
				props.epsilon,
				saved_mean.data,
				saved_var.data
			))

		} else {
			try cudaCheck(status: cudnnBatchNormalizationForwardInference(
				dataStream.cudnn.handle,
				bnMode.cudnn,
				inData.one,
				outData.zero,
				inTensor.desc,
				inData.ro(using: dataStream),
				outTensor.desc,
				outData.rw(using: dataStream),
				scaleBiasMeanVarianceTensor!.desc,
				scale.data,
				bias.data,
				running_mean.data,
				running_var.data,
				props.epsilon
			))
		}
	}
	
	//----------------------------------------------------------------------------
	// backward
	public func backward(outData: DataView, outGrad: DataView?,
	                     inData: DataView, inGrad: inout DataView?,
	                     solver: ModelSolver, labels: DataView?) throws {
		// inGrad is nil for inputs that don't perform backward
		if inGrad == nil { return }

		try cudaCheck(status: cudnnBatchNormalizationBackward(
			dataStream.cudnn.handle,
			bnMode.cudnn,
			inData.one,
			inData.zero,
			inData.one,
			inData.zero,
			inTensor.desc,
			inData.ro(using: dataStream),
			outTensor.desc,
			outGrad!.ro(using: dataStream),
			inTensor.desc,
			inGrad!.rw(using: dataStream),
			scaleBiasMeanVarianceTensor!.desc,
			scale.data,
			grad_scale.data,
			grad_bias.data,
			props.epsilon,
			saved_mean.data,
			saved_var.data
		))
	}

	//----------------------------------------------------------------------------
	// setupForward
	public func setupForward(mode: EvaluationMode, inData: DataView, labels: DataView?,
	                         outData: inout DataView, backData: inout DataView?) throws {
		// validate
		guard props.epsilon >= CUDNN_BN_MIN_EPSILON else {
			writeLog("\(props.namePath) epsilon must be greater than or equal to: " +
				"\(CUDNN_BN_MIN_EPSILON)")
			throw ModelError.setupFailed
		}

		// discover bnMode
		if props.mode == .auto {
			// TODO search upstream
			bnMode = BatchNormalizeMode.spatial
		} else {
			bnMode = props.mode
		}

		// assure the output is the correct type and size
		outData = DataView(shape: inData.shape, dataType: inData.dataType)

		// create tensor descriptors
		inTensor = try inData.createTensorDescriptor()
		outTensor = try outData.createTensorDescriptor()

		// create the scaleBiasMeanVarianceTensor descriptor
		var temp: cudnnTensorDescriptor_t?
		try cudaCheck(status: cudnnCreateTensorDescriptor(&temp))
		try cudaCheck(status: cudnnDeriveBNTensorDescriptor(temp!, inTensor.desc, bnMode.cudnn))
		scaleBiasMeanVarianceTensor = TensorDescriptor(owning: temp!)

		let (extent, strides, dataType) = try scaleBiasMeanVarianceTensor.getInfo()
		let shape = Shape(extent: extent, layout: .nchw, strides: strides)
		workspaceSize = shape.elementCount * dataType.size

		scale = try dataStream.device.createArray(count: workspaceSize)
		bias = try dataStream.device.createArray(count: workspaceSize)
		running_mean = try dataStream.device.createArray(count: workspaceSize)
		running_var = try dataStream.device.createArray(count: workspaceSize)

		if mode == .training {
			saved_mean = try dataStream.device.createArray(count: workspaceSize)
			saved_var = try dataStream.device.createArray(count: workspaceSize)
		}
	}

	//----------------------------------------------------------------------------
	// setupBackward
	public func setupBackward(outData: DataView, outGrad: DataView?, inData: DataView) throws {
		grad_scale = try dataStream.device.createArray(count: workspaceSize)
		grad_bias = try dataStream.device.createArray(count: workspaceSize)
	}
}

//==============================================================================
// Enum --> Cuda value mapping
//
extension BatchNormalizeMode {
	public var cudnn: cudnnBatchNormMode_t {
		get {
			switch self {
			case .auto: fatalError("BatchNormalizeMode auto is invalid case")
			case .perActivation: return CUDNN_BATCHNORM_PER_ACTIVATION
			case .spatial      : return CUDNN_BATCHNORM_SPATIAL
			}
		}
	}
}


