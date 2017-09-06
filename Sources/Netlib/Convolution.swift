//******************************************************************************
//  Created by Edward Connell on 4/11/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
final public class Convolution : ComputableFilterBase, ConvolutionProperties, InitHelper {
	//----------------------------------------------------------------------------
	// properties
	public var activationMode: ActivationMode?                                 { didSet{onSet("activationMode")} }
	public var activationNan = NanPropagation.noPropagate                      { didSet{onSet("activationNan")} }
	public var activationReluCeiling = 0.0                                     { didSet{onSet("activationReluCeiling")} }
	public var backwardDataAlgorithm = ConvolutionBwdDataAlgorithm.fastest     { didSet{onSet("backwardDataAlgorithm")} }
	public var backwardDataWorkspaceLimit = 10.MB                              { didSet{onSet("backwardDataWorkspaceLimit")} }
	public var backwardFilterAlgorithm = ConvolutionBwdFilterAlgorithm.fastest { didSet{onSet("backwardFilterAlgorithm")} }
	public var backwardFilterWorkspaceLimit = 10.MB                            { didSet{onSet("backwardFilterWorkspaceLimit")} }
	public var bias = LearnedParameter()                                       { didSet{onSet("bias")} }
	public var filterSize = [3]                                                { didSet{onSet("filterSize")} }
	public var forwardAlgorithm = ConvolutionFwdAlgorithm.fastest	             { didSet{onSet("forwardAlgorithm")} }
	public var forwardWorkspaceLimit = 10.MB	                                 { didSet{onSet("forwardWorkspaceLimit")} }
	public var mode = ConvolutionMode.crossCorrelation	                       { didSet{onSet("mode")} }
	public var outputChannels: Int?                                            { didSet{onSet("outputChannels")} }
	public var pad = [0]	                                                     { didSet{onSet("pad")} }
	public var stride = [1]                                                    { didSet{onSet("stride")} }
	public var dilation = [1]	                                                 { didSet{onSet("dilation")} }
	public var weights = LearnedParameter()                                    { didSet{onSet("weights")} }
	private var wsum, bsum: DataView!

	//----------------------------------------------------------------------------
	// addAccessors
	public override func addAccessors() {
		super.addAccessors()
		addAccessor(name: "activationMode",
			          get: { [unowned self] in self.activationMode },
			          set: { [unowned self] in self.activationMode = $0 })
		addAccessor(name: "activationNan",
			          get: { [unowned self] in self.activationNan },
			          set: { [unowned self] in self.activationNan = $0 })
		addAccessor(name: "activationReluCeiling",
			          get: { [unowned self] in self.activationReluCeiling },
			          set: { [unowned self] in self.activationReluCeiling = $0 })
		addAccessor(name: "backwardDataAlgorithm",
		            get: { [unowned self] in self.backwardDataAlgorithm },
		            set: { [unowned self] in self.backwardDataAlgorithm = $0 })
		addAccessor(name: "backwardDataWorkspaceLimit",
		            get: { [unowned self] in self.backwardDataWorkspaceLimit },
		            set: { [unowned self] in self.backwardDataWorkspaceLimit = $0 })
		addAccessor(name: "backwardFilterAlgorithm",
		            get: { [unowned self] in self.backwardFilterAlgorithm },
		            set: { [unowned self] in self.backwardFilterAlgorithm = $0 })
		addAccessor(name: "backwardFilterWorkspaceLimit",
		            get: { [unowned self] in self.backwardFilterWorkspaceLimit },
		            set: { [unowned self] in self.backwardFilterWorkspaceLimit = $0 })
		addAccessor(name: "bias", lookup: .noLookup,
		            get: { [unowned self] in self.bias },
		            set: { [unowned self] in self.bias = $0 })
		addAccessor(name: "dilation",
			          get: { [unowned self] in self.dilation },
			          set: { [unowned self] in self.dilation = $0 })
		addAccessor(name: "filterSize",
		            get: { [unowned self] in self.filterSize },
		            set: { [unowned self] in self.filterSize = $0 })
		addAccessor(name: "forwardAlgorithm",
		            get: { [unowned self] in self.forwardAlgorithm },
		            set: { [unowned self] in self.forwardAlgorithm = $0 })
		addAccessor(name: "forwardWorkspaceLimit",
		            get: { [unowned self] in self.forwardWorkspaceLimit },
		            set: { [unowned self] in self.forwardWorkspaceLimit = $0 })
		addAccessor(name: "mode",
		            get: { [unowned self] in self.mode },
		            set: { [unowned self] in self.mode = $0 })
		addAccessor(name: "outputChannels",
		            get: { [unowned self] in self.outputChannels },
		            set: { [unowned self] in self.outputChannels = $0 })
		addAccessor(name: "pad",
		            get: { [unowned self] in self.pad },
		            set: { [unowned self] in self.pad = $0 })
		addAccessor(name: "stride",
		            get: { [unowned self] in self.stride },
		            set: { [unowned self] in self.stride = $0 })
		addAccessor(name: "weights", lookup: .noLookup,
		            get: { [unowned self] in self.weights },
		            set: { [unowned self] in self.weights = $0 })
	}

	//----------------------------------------------------------------------------
	// onSetupData
	public override func onSetupData(using stream: DeviceStream) throws {
		try bias.setupData(dataType: outDataType, using: stream)
		try weights.setupData(dataType: outDataType, using: stream)
	}

	//----------------------------------------------------------------------------
	// onComputeLoss
	public override func onComputeLoss() throws -> Double {
		guard lossWeight > 0 else { return 0 }

		// lazy allocate
		if wsum == nil {
			let dataType = weights.data.dataType
			wsum = DataView(count: 1, dataType: dataType)
			bsum = DataView(count: 1, dataType: dataType)
		}

		// use the input stream
		let stream = try getInput().items[0].stream
		try stream.asum(x: weights.grad.flattened(), result: &wsum!)
		try stream.asum(x: bias.grad.flattened(), result: &bsum!)
		learningError = try wsum.get() + bsum.get()
		return learningError * lossWeight
	}
}

//==============================================================================
//
public protocol ConvolutionProperties : ComputableFilterProperties {
	var activationMode: ActivationMode? { get set }
	var activationNan: NanPropagation { get set }
	var activationReluCeiling: Double { get set }
	var backwardDataAlgorithm: ConvolutionBwdDataAlgorithm { get set }
	var backwardDataWorkspaceLimit: Int { get }
	var backwardFilterAlgorithm: ConvolutionBwdFilterAlgorithm { get set }
	var backwardFilterWorkspaceLimit: Int { get }
	var bias: LearnedParameter { get }
	var filterSize: [Int] { get }
	var weights: LearnedParameter { get }
	var forwardAlgorithm: ConvolutionFwdAlgorithm { get set }
	var forwardWorkspaceLimit: Int { get }
	var mode: ConvolutionMode { get }
	var outputChannels: Int? { get }
	var pad: [Int] { get }
	var stride: [Int] { get }
	var dilation: [Int] { get }
}

// ConvolutionFwdAlgorithm
public enum ConvolutionFwdAlgorithm : String, EnumerableType {
	case implicitGEMM
	case implicitPrecompGEMM
	case gemm
	case direct
	case fft
	case fftTiling
	case winograd
	case winogradNonFused
	case deterministic
	case fastest
	case noWorkspace
	case workspaceLimit
}

// ConvolutionBwdDataAlgorithm
public enum ConvolutionBwdDataAlgorithm : String, EnumerableType {
	case algo0
	case algo1
	case fft
	case fftTiling
	case winograd
	case winogradNonFused
	case deterministic
	case fastest
	case noWorkspace
	case workspaceLimit
}

// ConvolutionBwdFilterAlgorithm
public enum ConvolutionBwdFilterAlgorithm : String, EnumerableType {
	case algo0
	case algo1
	case algo3
	case fft
	//	case winograd // cudnn hasn't implemented yet
	case winogradNonFused
	case numAlgorithms
	case deterministic
	case fastest
	case noWorkspace
	case workspaceLimit
}

// ConvolutionMode
public enum ConvolutionMode : String, EnumerableType {
	case convolution
	case crossCorrelation
}

