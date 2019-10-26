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
final public class Softmax : ComputableFilterBase, SoftmaxProperties, InitHelper {
	//----------------------------------------------------------------------------
	// properties
	public var algorithm = SoftmaxAlgorithm.accurate { didSet{onSet("algorithm")} }
	public var mode = SoftmaxMode.channel            { didSet{onSet("mode")} }
	public var outputType = SoftmaxOutput.labels     { didSet{onSet("outputType")} }

	//----------------------------------------------------------------------------
	// addAccessors
	public override func addAccessors() {
		super.addAccessors()
		addAccessor(name: "algorithm",
		            get: { [unowned self] in self.algorithm },
		            set: { [unowned self] in self.algorithm = $0 })
		addAccessor(name: "mode",
		            get: { [unowned self] in self.mode },
		            set: { [unowned self] in self.mode = $0 })
		addAccessor(name: "outputType",
		            get: { [unowned self] in self.outputType },
		            set: { [unowned self] in self.outputType = $0 })
	}
}

//==============================================================================

public protocol SoftmaxProperties : ComputableFilterProperties {
	var algorithm: SoftmaxAlgorithm { get }
	var mode: SoftmaxMode { get }
	var outputType: SoftmaxOutput { get }
}

public enum SoftmaxAlgorithm : String, EnumerableType {
	case accurate, fast, log
}

public enum SoftmaxMode : String, EnumerableType {
	case channel, instance
}

public enum SoftmaxOutput : String, EnumerableType {
	case labels, probabilities
}

