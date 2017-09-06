//******************************************************************************
//  Created by Edward Connell on 4/11/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
final public class FullyConnected :
	ComputableFilterBase, FullyConnectedProperties, InitHelper {
	//----------------------------------------------------------------------------
	// properties
	public var outputChannels: Int?            { didSet{onSet("outputChannels")} }
	public var bias = LearnedParameter()       { didSet{onSet("bias")} }
	public var weights = LearnedParameter()    { didSet{onSet("weights")} }
	private var wsum, bsum: DataView!

	//----------------------------------------------------------------------------
	// addAccessors
	public override func addAccessors() {
		super.addAccessors()
		addAccessor(name: "bias", lookup: .noLookup,
		            get: { [unowned self] in self.bias },
		            set: { [unowned self] in self.bias = $0 })
		addAccessor(name: "outputChannels",
		            get: { [unowned self] in self.outputChannels },
		            set: { [unowned self] in self.outputChannels = $0 })
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
			wsum = DataView(count: 1, dataType: weights.data.dataType)
			bsum = DataView(count: 1, dataType: bias.data.dataType)
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
public protocol FullyConnectedProperties : ComputableFilterProperties {
	var outputChannels: Int? { get }
	var bias: LearnedParameter { get }
	var weights: LearnedParameter { get }
}









