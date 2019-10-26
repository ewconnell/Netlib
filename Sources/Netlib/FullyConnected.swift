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









