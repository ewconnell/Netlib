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

public class CudaConvolution : Computable {
	// initializers
	public init(log: Log?, props: ComputableFilterProperties, stream: CudaStream) {
		self.props = (props as? ConvolutionProperties)!
		currentLog = log
		dataStream = stream
		trackingId = objectTracker.register(type: self)
	}
	deinit { objectTracker.remove(trackingId: trackingId) }

	//----------------------------------------------------------------------------
	// properties
	public private(set) var trackingId = 0
	private weak var props: ConvolutionProperties!

	// logging
	public var logLevel = LogLevel.error
	public var nestingLevel = 0
	public weak var currentLog: Log?

	// streams
	public  var stream: DeviceStream { return dataStream }
	private let dataStream: CudaStream

	// descriptors
	private var inTensor: TensorDescriptor!
	private var outTensor: TensorDescriptor!
	private var biasTensor: TensorDescriptor!
	private var filterDesc: FilterDescriptor!
	private var convolutionDesc: ConvolutionDescriptor!
	private var activationDesc: ActivationDescriptor!

	// forward
	private var fwdAlgo: cudnnConvolutionFwdAlgo_t!
	private var fwdWorkspaceSize = 0
	private var fwdWorkspace: DeviceArray?

	// backward data
	private var bwdDataAlgo: cudnnConvolutionBwdDataAlgo_t!
	private var bwdDataWorkspaceSize = 0
	private var bwdDataWorkspace: DeviceArray?

	// backward filter
	private var bwdFilterAlgo: cudnnConvolutionBwdFilterAlgo_t!
	private var bwdFilterWorkspaceSize = 0
	private var bwdFilterWorkspace: DeviceArray?
	
	//----------------------------------------------------------------------------
	// forward
	public func forward(mode: EvaluationMode, inData: DataView, labels: DataView?,
	                    outData: inout DataView, backData: inout DataView?) throws {
		if activationDesc == nil {
			try cudaCheck(status: cudnnConvolutionForward(
				dataStream.cudnn.handle,
				inData.one,
				inTensor.desc,
				inData.ro(using: dataStream),
				filterDesc.desc,
				props.weights.data.ro(using: dataStream),
				convolutionDesc.desc,
				fwdAlgo,
				fwdWorkspace?.data,
				fwdWorkspaceSize,
				outData.zero,
				outTensor.desc,
				outData.rw(using: dataStream)))

			// add bias
			try cudaCheck(status: cudnnAddTensor(
				dataStream.cudnn.handle,
				props.bias.data.one,
				biasTensor.desc,
				props.bias.data.ro(using: dataStream),
				outData.one,
				outTensor.desc,
				outData.rw(using: dataStream)))
		} else {
			// fused conv/bias/act
			try cudaCheck(status: cudnnConvolutionBiasActivationForward(
				dataStream.cudnn.handle,
				inData.one,
				inTensor.desc,
				inData.ro(using: dataStream),
				filterDesc.desc,
				props.weights.data.ro(using: dataStream),
				convolutionDesc.desc,
				fwdAlgo,
				fwdWorkspace?.data,
				fwdWorkspaceSize,
				outData.zero,
				outTensor.desc,
				outData.ro(using: dataStream),
				biasTensor.desc,
				props.bias.data.ro(using: dataStream),
				activationDesc.desc,
				outTensor.desc,
				outData.rw(using: dataStream)))
		}
	}
	
	//----------------------------------------------------------------------------
	// backward
	public func backward(outData: DataView, outGrad: DataView?,
	                     inData: DataView, inGrad: inout DataView?,
	                     solver: ModelSolver, labels: DataView?) throws {
		// data
		// inGrad is nil for inputs that don't perform backward
		if inGrad != nil {
			try cudaCheck(status: cudnnConvolutionBackwardData(
				dataStream.cudnn.handle,
				props.weights.data.one,
				filterDesc.desc,
				props.weights.data.ro(using: dataStream),
				outTensor.desc,
				outGrad!.ro(using: dataStream),
				convolutionDesc.desc,
				bwdDataAlgo,
				bwdDataWorkspace?.data,
				bwdDataWorkspaceSize,
				inGrad!.zero,
				inTensor.desc,
				inGrad!.rw(using: dataStream)))
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

		//----------------------------------
		// setup filter weights
		let convolutionRank = inData.extent.count - 2
		
		let filterExtent = [outputChannels, inData.channels] +
			expand(array: props.filterSize, to: convolutionRank)
		
		try props.weights.setExtent(filterExtent, using: dataStream)
		
		filterDesc = try FilterDescriptor(shape: props.weights.data.shape,
			                                dataType: props.weights.dataType)

		//----------------------------------
		// setup filter bias
		try props.bias.setExtent([1, outputChannels, 1, 1], using: dataStream)
		biasTensor = try props.bias.data.createTensorDescriptor()

		//----------------------------------
		// setup convolution
		let stride = expand(array: props.stride, to: convolutionRank)
		let pad = expand(array: props.pad, to: convolutionRank)
		let dilation = expand(array: props.dilation, to: convolutionRank)
		let convDataType:DataType = inData.dataType == .real64F ? .real64F : .real32F

		convolutionDesc = try ConvolutionDescriptor(
			dataType: convDataType,	rank: convolutionRank,
			pad: pad, stride: stride, dilation: dilation, mode: props.mode)
		
		//----------------------------------
		// get the dims for the output
		inTensor = try inData.createTensorDescriptor()
		var outputDims = [Int32](repeating: 0, count: inData.extent.count)
		
		try cudaCheck(status: cudnnGetConvolutionNdForwardOutputDim(
			convolutionDesc.desc, inTensor.desc, filterDesc.desc,
			Int32(inData.extent.count), &outputDims))

		// initialize output buffer and tensor
		outData = DataView(shape: Shape(extent: outputDims.map { Int($0) }),
			                 dataType: props.outDataType)
		outTensor = try outData.createTensorDescriptor()

		//----------------------------------
		// choose best forward algorithm
		switch props.forwardAlgorithm {
		case .deterministic:
			let algs = try findForwardAlgorithms(inData: inData)
			var notFound = true
			for alg in algs {
				if alg.determinism == CUDNN_DETERMINISTIC {
					notFound = false
					fwdAlgo = alg.algo
					fwdWorkspaceSize = alg.memory
					break
				}
			}

			// default to the fastest
			if notFound {
				fwdAlgo = algs[0].algo
				fwdWorkspaceSize = algs[0].memory
				writeLog("failed to find 'deterministic' forward convolution " +
					"algorithm. 'fastest' used instead")
			}

		case .fastest:
			let algs = try findForwardAlgorithms(inData: inData)
			fwdAlgo = algs[0].algo
			fwdWorkspaceSize = algs[0].memory

		case .noWorkspace:
			let algs = try findForwardAlgorithms(inData: inData)
			var algIndex = -1
			for i in 0..<algs.count {
				if algs[i].memory == 0 { algIndex = i; break }
			}
			
			guard algIndex >= 0 else {
				props.writeLog("failed to find 'noWorkspace' forward convolution algorithm")
				throw ModelError.setupFailed
			}
			fwdAlgo = algs[algIndex].algo
			fwdWorkspaceSize = algs[algIndex].memory
			
		case .workspaceLimit:
			let algs = try findForwardAlgorithms(inData: inData)
			var algIndex = -1
			for i in 0..<algs.count {
				if algs[i].memory <= props.forwardWorkspaceLimit { algIndex = i; break }
			}
			
			guard algIndex >= 0 else {
				props.writeLog("failed to find suitable 'workspaceLimit' " +
					"forward convolution algorithm")
				throw ModelError.setupFailed
			}
			fwdAlgo = algs[algIndex].algo
			fwdWorkspaceSize = algs[algIndex].memory
			
		default:
			// user explicitly specifies
			fwdAlgo = props.forwardAlgorithm.cudnn

			// get the workspace size
			try cudaCheck(status: cudnnGetConvolutionForwardWorkspaceSize(
				dataStream.cudnn.handle,
				inTensor.desc,
				filterDesc.desc,
				convolutionDesc.desc,
				outTensor.desc,
				fwdAlgo,
				&fwdWorkspaceSize))
		}
		
		// allocate workspace
		if fwdWorkspaceSize > 0 {
			fwdWorkspace = try dataStream.device.createArray(count: fwdWorkspaceSize)
		}

		// report selection
		let alg = ConvolutionFwdAlgorithm(cudnn: fwdAlgo)

		if props.willLog(level: .diagnostic) && props.forwardAlgorithm != alg {
			props.diagnostic("\(props.namePath) using forward algorithm: " +
				"\(alg)  workspace size: \(fwdWorkspaceSize)",
				categories: [.setup, .setupForward])
		}

		// write back if changed
		// don't always assign or else it will trigger a property change event
		if props.forwardAlgorithm != alg { props.forwardAlgorithm = alg }

		// setup optional activation
		if let mode = props.activationMode {
			activationDesc = try ActivationDescriptor(
				mode: mode, nan: props.activationNan,
				reluCeiling: props.activationReluCeiling)
		}
	}
	
	//----------------------------------------------------------------------------
	// setupBackward
	public func setupBackward(outData: DataView, outGrad: DataView?, inData: DataView) throws {
		//----------------------------------
		// set weights gradient update function
		let weightsStream = try dataStream.device
			.createStream(label: "\(props.namespaceName).weightsStream")

		props.weights.setGradientUpdateFunction(using: weightsStream) { [unowned self] in
			let stream = $0 as! CudaStream
			try cudaCheck(status: cudnnConvolutionBackwardFilter(
				stream.cudnn.handle,
				inData.one,
				self.inTensor.desc,
				inData.ro(using: stream),
				self.outTensor.desc,
				outGrad!.ro(using: stream),
				self.convolutionDesc.desc,
				self.bwdFilterAlgo,
				self.bwdFilterWorkspace?.data,
				self.bwdFilterWorkspaceSize,
				self.props.weights.grad.zero,
				self.filterDesc.desc,
				self.props.weights.grad.rw(using: stream)))
		}

		//----------------------------------
		// set bias gradient update function
		let biasStream = try dataStream.device
			.createStream(label: "\(props.namespaceName).biasStream") as! CudaStream

		props.bias.setGradientUpdateFunction(using: biasStream) { [unowned self] in
			let stream = $0 as! CudaStream
			try cudaCheck(status: cudnnConvolutionBackwardBias(
				stream.cudnn.handle,
				outGrad!.one,
				self.outTensor.desc,
				outGrad!.ro(using: stream),
				self.props.bias.grad.zero,
				self.biasTensor.desc,
				self.props.bias.grad.rw(using: stream)))
		}

		// weights
		let dataType = props.weights.data.dataType
		let weights  = props.weights
		weights.grad = DataView(shape: weights.data.shape, dataType: dataType)

		// bias
		let bias = props.bias
		bias.grad = DataView(shape: bias.data.shape, dataType: dataType)

		//----------------------------------
		// choose best backward data algorithm
		switch props.backwardDataAlgorithm {
		case .deterministic:
			let algs = try findBackwardDataAlgorithms(outData: outData, inData: inData)
			var notFound = true
			for alg in algs {
				if alg.determinism == CUDNN_DETERMINISTIC {
					notFound = false
					bwdDataAlgo = alg.algo
					bwdDataWorkspaceSize = alg.memory
					break
				}
			}

			// default to the fastest
			if notFound {
				bwdDataAlgo = algs[0].algo
				bwdDataWorkspaceSize = algs[0].memory
				writeLog("failed to find 'deterministic' backward data convolution " +
					"algorithm. 'fastest' used instead")
			}

		case .fastest:
			let algs = try findBackwardDataAlgorithms(outData: outData, inData: inData)
			bwdDataAlgo = algs[0].algo
			bwdDataWorkspaceSize = algs[0].memory
			
		case .noWorkspace:
			let algs = try findBackwardDataAlgorithms(outData: outData, inData: inData)
			var algIndex = -1
			for i in 0..<algs.count {
				if algs[i].memory == 0 { algIndex = i; break }
			}
			
			guard algIndex >= 0 else {
				props.writeLog("failed to find 'noWorkspace' backward data convolution algorithm")
				throw ModelError.setupFailed
			}
			bwdDataAlgo = algs[algIndex].algo
			bwdDataWorkspaceSize = algs[algIndex].memory
			
		case .workspaceLimit:
			let algs = try findBackwardDataAlgorithms(outData: outData, inData: inData)
			var algIndex = -1
			for i in 0..<algs.count {
				if algs[i].memory <= props.backwardDataWorkspaceLimit { algIndex = i; break }
			}
			
			guard algIndex >= 0 else {
				props.writeLog("failed to find suitable 'workspaceLimit' " +
					"backward data convolution algorithm")
				throw ModelError.setupFailed
			}
			bwdDataAlgo = algs[algIndex].algo
			bwdDataWorkspaceSize = algs[algIndex].memory
			
		default:
			// user explicitly specifies
			bwdDataAlgo = props.backwardDataAlgorithm.cudnn
			
			// get the workspace size
			try cudaCheck(status: cudnnGetConvolutionBackwardDataWorkspaceSize(
				dataStream.cudnn.handle,
				filterDesc.desc,
				outTensor.desc,
				convolutionDesc.desc,
				inTensor.desc,
				bwdDataAlgo,
				&bwdDataWorkspaceSize))
		}
		
		// allocate workspace
		if bwdDataWorkspaceSize > 0 {
			bwdDataWorkspace =
				try dataStream.device.createArray(count: bwdDataWorkspaceSize)
		}

		// report selection
		let dataAlg = ConvolutionBwdDataAlgorithm(cudnn: bwdDataAlgo)

		if props.willLog(level: .diagnostic) && props.backwardDataAlgorithm != dataAlg {
			props.diagnostic("\(props.namePath) using backward data algorithm: " +
				"\(dataAlg)  workspace size: \(bwdDataWorkspaceSize)",
				categories: [.setup, .setupBackward])
		}

		// write back if changed
		// don't just assign or else it will trigger a change event
		if props.backwardDataAlgorithm != dataAlg {
			props.backwardDataAlgorithm = dataAlg
		}

		//----------------------------------
		// choose best backward filter algorithm
		switch props.backwardFilterAlgorithm {
		case .deterministic:
			let algs = try findBackwardFilterAlgorithms(outData: outData, inData: inData)
			var notFound = true
			for alg in algs {
				if alg.determinism == CUDNN_DETERMINISTIC {
					notFound = false
					bwdFilterAlgo = alg.algo
					bwdFilterWorkspaceSize = alg.memory
					break
				}
			}

			// default to the fastest
			if notFound {
				bwdFilterAlgo = algs[0].algo
				bwdFilterWorkspaceSize = algs[0].memory
				writeLog("failed to find 'deterministic' backward filter convolution " +
					"algorithm. 'fastest' used instead")
			}

		case .fastest:
			let algs = try findBackwardFilterAlgorithms(outData: outData, inData: inData)
			bwdFilterAlgo = algs[0].algo
			bwdFilterWorkspaceSize = algs[0].memory
			
		case .noWorkspace:
			let algs = try findBackwardFilterAlgorithms(outData: outData, inData: inData)
			var algIndex = -1
			for i in 0..<algs.count {
				if algs[i].memory == 0 { algIndex = i; break }
			}
			
			guard algIndex >= 0 else {
				props.writeLog("failed to find 'noWorkspace' backward filter convolution algorithm")
				throw ModelError.setupFailed
			}
			bwdFilterAlgo = algs[algIndex].algo
			bwdFilterWorkspaceSize = algs[algIndex].memory
			
		case .workspaceLimit:
			let algs = try findBackwardFilterAlgorithms(outData: outData, inData: inData)
			var algIndex = -1
			for i in 0..<algs.count {
				if algs[i].memory <= props.backwardFilterWorkspaceLimit { algIndex = i; break }
			}
			
			guard algIndex >= 0 else {
				props.writeLog("failed to find suitable 'workspaceLimit' " +
					"backward filter convolution algorithm")
				throw ModelError.setupFailed
			}
			bwdFilterAlgo = algs[algIndex].algo
			bwdFilterWorkspaceSize = algs[algIndex].memory
			
		default:
			// user explicitly specifies
			bwdFilterAlgo = props.backwardFilterAlgorithm.cudnn
			
			// get the workspace size
			try cudaCheck(status: cudnnGetConvolutionBackwardFilterWorkspaceSize(
				dataStream.cudnn.handle,
				inTensor.desc,
				outTensor.desc,
				convolutionDesc.desc,
				filterDesc.desc,
				bwdFilterAlgo,
				&bwdFilterWorkspaceSize))
		}
		
		// allocate workspace
		if bwdFilterWorkspaceSize > 0 {
			bwdFilterWorkspace =
				try props.weights.updateStream.device.createArray(count: bwdFilterWorkspaceSize)
		}

		// report selection
		let filterAlg = ConvolutionBwdFilterAlgorithm(cudnn: bwdFilterAlgo)

		if props.willLog(level: .diagnostic) && props.backwardFilterAlgorithm != filterAlg {
			props.diagnostic("\(props.namePath) using backward filter algorithm: " +
				"\(filterAlg)  workspace size: \(bwdFilterWorkspaceSize)",
				categories: [.setup, .setupBackward])
		}

		// write back if changed
		// don't just assign or else it will trigger a change event
		if props.backwardFilterAlgorithm != filterAlg {
			props.backwardFilterAlgorithm = filterAlg
		}
	}
	
	//----------------------------------------------------------------------------
	// findForwardAlgorithms
	private func findForwardAlgorithms(inData: DataView) throws
		-> [cudnnConvolutionFwdAlgoPerf_t] {
		// get the list of forward algorithms
		var returnedAlgoCount: Int32 = 0
		var results = [cudnnConvolutionFwdAlgoPerf_t](
			repeating: cudnnConvolutionFwdAlgoPerf_t(),
			count: numConvolutionFwdAlgorithms)

		try cudaCheck(status: cudnnFindConvolutionForwardAlgorithm(
			dataStream.cudnn.handle,
			inTensor.desc,
			filterDesc.desc,
			convolutionDesc.desc,
			outTensor.desc,
			Int32(results.count),
			&returnedAlgoCount,
			&results))

		// report
		if props.willLog(level: .diagnostic) {
			let cat: LogCategories = [.setup, .setupForward]
			props.diagnostic("", categories: cat)
			props.diagnostic("find forward algorithms", categories: cat, trailing: "-")

			for item in results {
				let alg = ConvolutionFwdAlgorithm(cudnn: item.algo)
				let det = item.determinism == CUDNN_DETERMINISTIC ?
					"deterministic" : "non-deterministic"
				props.diagnostic("Algorithm: \(alg)  time: \(item.time) " +
					"required memory: \(item.memory)  \(det)", categories: cat)
			}
		}
		
		results.removeLast(results.count - Int(returnedAlgoCount))
		return results
	}
	
	//----------------------------------------------------------------------------
	// findBackwardDataAlgorithms
	private func findBackwardDataAlgorithms(outData: DataView, inData: DataView)
		throws -> [cudnnConvolutionBwdDataAlgoPerf_t] {
		// get the list of forward algorithms
		var returnedAlgoCount: Int32 = 0
		var results = [cudnnConvolutionBwdDataAlgoPerf_t](
			repeating: cudnnConvolutionBwdDataAlgoPerf_t(),
			count: numConvolutionBwdDataAlgorithms)

		try cudaCheck(status: cudnnFindConvolutionBackwardDataAlgorithm(
			dataStream.cudnn.handle,
			filterDesc.desc,
			outTensor.desc,
			convolutionDesc.desc,
			inTensor.desc,
			Int32(results.count),
			&returnedAlgoCount,
			&results))
		
		if props.willLog(level: .diagnostic) {
			let cat: LogCategories = [.setup, .setupBackward]
			props.diagnostic("", categories: cat)
			props.diagnostic("find backward data algorithms", categories: cat, trailing: "-")
			
			for item in results {
				let alg = ConvolutionBwdDataAlgorithm(cudnn: item.algo)
				let det = item.determinism == CUDNN_DETERMINISTIC ?
					"deterministic" : "non-deterministic"
				props.diagnostic("Algorithm: \(alg)  time: \(item.time) " +
					"required memory: \(item.memory)  \(det)", categories: cat)
			}
		}
		
		results.removeLast(results.count - Int(returnedAlgoCount))
		return results
	}
	
	//----------------------------------------------------------------------------
	// findBackwardFilterAlgorithms
	private func findBackwardFilterAlgorithms(outData: DataView, inData: DataView)
		throws -> [cudnnConvolutionBwdFilterAlgoPerf_t] {
		// get the list of forward algorithms
		var returnedAlgoCount: Int32 = 0
		var results = [cudnnConvolutionBwdFilterAlgoPerf_t](
			repeating: cudnnConvolutionBwdFilterAlgoPerf_t(),
			count: numConvolutionBwdFilterAlgorithms)
		
		try cudaCheck(status: cudnnFindConvolutionBackwardFilterAlgorithm(
			dataStream.cudnn.handle,
			inTensor.desc,
			outTensor.desc,
			convolutionDesc.desc,
			filterDesc.desc,
			Int32(results.count),
			&returnedAlgoCount,
			&results))
		
		if props.willLog(level: .diagnostic) {
			let cat: LogCategories = [.setup, .setupBackward]
			props.diagnostic("", categories: cat)
			props.diagnostic("find backward filter algorithms", categories: cat, trailing: "-")
			
			for item in results {
				let alg = ConvolutionBwdFilterAlgorithm(cudnn: item.algo)
				let det = item.determinism == CUDNN_DETERMINISTIC ?
					"deterministic" : "non-deterministic"
				props.diagnostic("Algorithm: \(alg)  time: \(item.time) " +
					"required memory: \(item.memory)  \(det)", categories: cat)
			}
		}
		
		results.removeLast(results.count - Int(returnedAlgoCount))
		return results
	}
}

//==============================================================================
// Enum --> Cuda value mapping
//
extension ConvolutionFwdAlgorithm {
	public var cudnn: cudnnConvolutionFwdAlgo_t {
		get {
			switch self {
			case .implicitGEMM       : return CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM
			case .implicitPrecompGEMM: return CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
			case .gemm               : return CUDNN_CONVOLUTION_FWD_ALGO_GEMM
			case .direct             : return CUDNN_CONVOLUTION_FWD_ALGO_DIRECT
			case .fft                : return CUDNN_CONVOLUTION_FWD_ALGO_FFT
			case .fftTiling          : return CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING
			case .winograd           : return CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD
			case .winogradNonFused   : return CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED
			default: fatalError("Invalid state")
			}
		}
	}
	
	public init(cudnn: cudnnConvolutionFwdAlgo_t) {
		switch cudnn {
		case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM        : self = .implicitGEMM
		case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM: self = .implicitPrecompGEMM
		case CUDNN_CONVOLUTION_FWD_ALGO_GEMM                 : self = .gemm
		case CUDNN_CONVOLUTION_FWD_ALGO_DIRECT               : self = .direct
		case CUDNN_CONVOLUTION_FWD_ALGO_FFT                  : self = .fft
		case CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING           : self = .fftTiling
		case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD             : self = .winograd
		case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED    : self = .winogradNonFused
		default: fatalError("Invalid state")
		}
	}
}

let numConvolutionFwdAlgorithms = 8

extension ConvolutionBwdDataAlgorithm {
	public var cudnn: cudnnConvolutionBwdDataAlgo_t {
		get {
			switch self {
			case .algo0           : return CUDNN_CONVOLUTION_BWD_DATA_ALGO_0
			case .algo1           : return CUDNN_CONVOLUTION_BWD_DATA_ALGO_1
			case .fft             : return CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT
			case .fftTiling       : return CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING
			case .winograd        : return CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD
			case .winogradNonFused: return CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED
			default: fatalError("ConvolutionBwdDataAlgorithm unknown value")
			}
		}
	}
	
	public init(cudnn: cudnnConvolutionBwdDataAlgo_t) {
		switch cudnn {
		case CUDNN_CONVOLUTION_BWD_DATA_ALGO_0                : self = .algo0
		case CUDNN_CONVOLUTION_BWD_DATA_ALGO_1                : self = .algo1
		case CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT              : self = .fft
		case CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING       : self = .fftTiling
		case CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD         : self = .winograd
		case CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED: self = .winogradNonFused
		default: fatalError("ConvolutionBwdDataAlgorithm unknown value")
		}
	}
}

let numConvolutionBwdDataAlgorithms = 6

extension ConvolutionBwdFilterAlgorithm {
	public var cudnn: cudnnConvolutionBwdFilterAlgo_t {
		get {
			switch self {
			case .algo0           : return CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0
			case .algo1           : return CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1
			case .algo3           : return CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3
			case .fft             : return CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT
//			case .winograd        : return CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD // cudnn not implemented yet
			case .winogradNonFused: return CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED
			default: fatalError("ConvolutionBwdFilterAlgorithm unknown value")
			}
		}
	}
	
	public init(cudnn: cudnnConvolutionBwdFilterAlgo_t) {
		switch cudnn {
		case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0                : self = .algo0
		case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1                : self = .algo1
		case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3                : self = .algo3
		case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT              : self = .fft
//					case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD: self = .winograd
		case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED: self = .winogradNonFused
		default: fatalError("ConvolutionBwdFilterAlgorithm unknown value")
		}
	}
}

let numConvolutionBwdFilterAlgorithms = 5

extension ConvolutionMode {
	public var cudnn: cudnnConvolutionMode_t {
		get {
			switch self {
			case .convolution     : return CUDNN_CONVOLUTION
			case .crossCorrelation: return CUDNN_CROSS_CORRELATION
			}
		}
	}
}


