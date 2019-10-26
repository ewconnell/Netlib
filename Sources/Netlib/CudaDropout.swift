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
//
//  https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf
//
// TODO: Choose implementation
// my version seems to generate better training results,
// uses 1/25 the gpu memory, and is faster on Maxwell Titan X
// mine:  F ~17.2 micro sec, B: ~2.4
// cudnn: F ~21 micro sec, B: ~ 3.4


import Cuda
import CudaKernels

#if true

public class CudaDropout : Computable {
	// initializers
	public init(log: Log?, props: ComputableFilterProperties, stream: CudaStream) {
		self.props = (props as? DropoutProperties)!
		currentLog = log
		dataStream = stream
		trackingId = objectTracker.register(type: self)
	}
	deinit { objectTracker.remove(trackingId: trackingId) }

	//----------------------------------------------------------------------------
	// properties
	public private(set) var trackingId = 0
	public var stream: DeviceStream { return dataStream }
	private weak var props: DropoutProperties!
	private let dataStream: CudaStream
	private var generatorState: RandomGeneratorState!
	private var mask: DataView!

	// logging
	public var logLevel = LogLevel.error
	public var nestingLevel = 0
	public weak var currentLog: Log?

	//----------------------------------------------------------------------------
	// forward
	//  Note: if we are not training then do nothing
	public func forward(mode: EvaluationMode, inData: DataView, labels: DataView?,
	                    outData: inout DataView, backData: inout DataView?) throws {

		if mode == .training {
			let flatInData = inData.flattened()
			var flatOutData = try outData.referenceFlattened(using: dataStream)

			try cudaCheck(status: cudaDropoutForward(
				cudaDataShape(from: flatInData), flatInData.ro(using: dataStream),
				cudaDataShape(from: flatOutData), flatOutData.rw(using: dataStream),
				props.drop,
				mask.rw(using: dataStream),
				(generatorState as! CudaRandomGeneratorState).handle,
				dataStream.handle))

		} else {
			outData = inData
		}
	}

	//----------------------------------------------------------------------------
	// backward
	public func backward(outData: DataView, outGrad: DataView?,
	                     inData: DataView, inGrad: inout DataView?,
	                     solver: ModelSolver, labels: DataView?) throws {
		// inGrad is nil for inputs that don't perform backward
		if inGrad == nil { return }
		var inGradRef = try inGrad!.referenceFlattened(using: dataStream)

		try cudaCheck(status: cudaDropoutBackward(
			cudaDataShape(from: outGrad!.flattened()), outGrad!.ro(using: dataStream),
			cudaDataShape(from: inGradRef), inGradRef.rw(using: dataStream),
			mask.ro(using: dataStream),
			dataStream.handle))
	}

	//----------------------------------------------------------------------------
	// setupForward
	public func setupForward(mode: EvaluationMode, inData: DataView, labels: DataView?,
	                         outData: inout DataView, backData: inout DataView?) throws {
		if mode == .training {
			// create generator state
			generatorState =
				try dataStream.createRandomGeneratorState(for: inData, seed: props.seed)

			// create mask buffer
			mask = DataView(count: inData.elementCount, dataType: inData.dataType)

			// create output
			outData = DataView(shape: inData.shape, dataType: inData.dataType)
		}
	}
}

#else
public class CudaDropout : Computable {
	// initializers
	public init(log: Log?, props: ComputableFilterProperties, stream: CudaStream) {
		self.props = props as! DropoutProperties
		currentLog = log
		dataStream = stream
		trackingId = objectTracker.register(type: self)
	}
	deinit { objectTracker.remove(trackingId: trackingId) }

	//----------------------------------------------------------------------------
	// properties
	public private(set) var trackingId = 0
	public var stream: DeviceStream { return dataStream }
	private weak var props: DropoutProperties!
	private let dataStream: CudaStream
	private var tensorDesc: TensorDescriptor!
	private var workspace: DeviceArray!
	private var dropoutDesc: DropoutDescriptor!

	// logging
	public var logLevel = LogLevel.error
	public var nestingLevel = 0
	public weak var currentLog: Log?

	//----------------------------------------------------------------------------
	// forward
	//  Note: if we are not training then do nothing
	public func forward(mode: EvaluationMode, inData: DataView, labels: DataView?,
	                    outData: inout DataView, backData: inout DataView?) throws {

		if mode == .training {
			try cudaCheck(status: cudnnDropoutForward(
				dataStream.cudnn.handle,
				dropoutDesc.desc,
				tensorDesc.desc,
				inData.ro(using: dataStream),
				tensorDesc.desc,
				outData.rw(using: dataStream),
				workspace.data,
				workspace.count
			))

		} else {
			outData = inData
		}
	}

	//----------------------------------------------------------------------------
	// backward
	public func backward(outData: DataView, outGrad: DataView?,
	                     inData: DataView, inGrad: inout DataView?,
	                     solver: ModelSolver, labels: DataView?) throws {
		// inGrad is nil for inputs that don't perform backward
		if inGrad == nil { return }
		try cudaCheck(status: cudnnDropoutBackward(
			dataStream.cudnn.handle,
			dropoutDesc.desc,
			tensorDesc.desc,
			outGrad!.ro(using: dataStream),
			tensorDesc.desc,
			inGrad!.rw(using: dataStream),
			workspace.data,
			workspace.count
		))
	}
	
	//----------------------------------------------------------------------------
	// setupForward
	public func setupForward(mode: EvaluationMode, inData: DataView, labels: DataView?,
	                         outData: inout DataView, backData: inout DataView?) throws {
		assert(inData.shape.isContiguous)

		if mode == .training {
			// create tensor descriptor
			let flatInData = inData.flattened(axis: 1)
			let extent = [flatInData.rows, 1, 1, flatInData.cols]
			let layout = inData.layout == .matrix ? .nchw : inData.layout
			let tensorShape = Shape(extent: extent, layout: layout)
			tensorDesc = try inData.createTensorDescriptor(asShape: tensorShape)

			// create output buffer
			outData = DataView(shape: inData.shape, dataType: inData.dataType)

			// create descriptor
			dropoutDesc = try DropoutDescriptor(
				stream: dataStream,
				drop: props.drop,
				seed: UInt64(props.seed ?? 0),
				tensorDesc: tensorDesc)

			// create workspace
			var sizeInBytes = 0
			try cudaCheck(status: cudnnDropoutGetReserveSpaceSize(tensorDesc.desc, &sizeInBytes))

			workspace = try stream.device.createArray(count: sizeInBytes)
		}
	}
}
#endif

