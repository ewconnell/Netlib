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
import Cuda

public final class CudaBatchNormalize : Computable {
	// initializers
	public init(log: Log?, props: ComputableFilterProperties, stream: CudaStream) {
		self.props = (props as? BatchNormalizeProperties)!
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
	private var saved_mean: DeviceArray!
	private var saved_var: DeviceArray!
	private var grad_scale: DeviceArray!
	private var grad_bias: DeviceArray!
	private var scale: DataView!
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
				scale.ro(using: dataStream),
				bias.data,
				expAverageFactor,
				props.running_mean?.rw(using: dataStream),
				props.running_var?.rw(using: dataStream),
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
				scale.ro(using: dataStream),
				bias.data,
				props.running_mean?.ro(using: dataStream),
				props.running_var?.ro(using: dataStream),
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
			scale.ro(using: dataStream),
			grad_scale.data,
			grad_bias.data,
			props.epsilon,
			saved_mean.data,
			saved_var.data
		))
	}

	//----------------------------------------------------------------------------
	// createZeroWorkspace
	private func createZeroWorkspace(count: Int) throws -> DeviceArray {
		let array = try dataStream.device.createArray(count: count)
		try array.zero(using: dataStream)
		return array
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

		var ones = DataView(shape: shape, dataType: dataType)
		try dataStream.fill(data: &ones, with: 1.0)
		scale = ones
		bias = try createZeroWorkspace(count: workspaceSize)

		props.running_mean = DataView(shape: shape, dataType: dataType)
		props.running_var = DataView(shape: shape, dataType: dataType)

		if mode == .training {
			saved_mean = try createZeroWorkspace(count: workspaceSize)
			saved_var = try createZeroWorkspace(count: workspaceSize)
		}
	}

	//----------------------------------------------------------------------------
	// setupBackward
	public func setupBackward(outData: DataView, outGrad: DataView?, inData: DataView) throws {
		grad_scale = try createZeroWorkspace(count: workspaceSize)
		grad_bias = try createZeroWorkspace(count: workspaceSize)
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


