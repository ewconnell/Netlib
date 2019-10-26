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

public class CudaLrnCrossChannel : Computable {
	// initializers
	public init(log: Log?, props: ComputableFilterProperties, stream: CudaStream) {
		self.props = (props as? LrnCrossChannelProperties)!
		currentLog = log
		dataStream = stream
		trackingId = objectTracker.register(type: self)
	}

	deinit { objectTracker.remove(trackingId: trackingId) }

	//----------------------------------------------------------------------------
	// properties
	public private(set) var trackingId = 0
	public var stream: DeviceStream { return dataStream }
	private weak var props: LrnCrossChannelProperties!
	private var lrnDescriptor: LRNDescriptor!
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

		try cudaCheck(status: cudnnLRNCrossChannelForward(
			dataStream.cudnn.handle,
			lrnDescriptor.desc,
			props.mode.cudnn,
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

		try cudaCheck(status: cudnnLRNCrossChannelBackward(
			dataStream.cudnn.handle,
			lrnDescriptor.desc,
			props.mode.cudnn,
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
		lrnDescriptor = try LRNDescriptor(N: props.windowSize, alpha: props.alpha,
		                                  beta: props.beta, K: props.k)
		
		// assure the output is the correct type and size
		outData = DataView(shape: inData.shape, dataType: inData.dataType)
		
		// create tensor descriptors
		assert(inData.layout == .matrix)
		let tensorShape = Shape(extent: [inData.rows, inData.cols, 1, 1], layout: .nchw)
		inTensor  = try inData.createTensorDescriptor(asShape: tensorShape)
		outTensor = try outData.createTensorDescriptor(asShape: tensorShape)
	}
}

//==============================================================================
// Enum --> Cuda value mapping
extension LrnCrossChannelMode {
	public var cudnn: cudnnLRNMode_t {
		get {
			switch self {
			case .crossChannelDim1: return CUDNN_LRN_CROSS_CHANNEL_DIM1
			}
		}
	}
}


