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
public final class Activation : ComputableFilterBase, ActivationProperties, InitHelper {
	//----------------------------------------------------------------------------
	// properties
	public var mode = ActivationMode.relu         { didSet{onSet("mode")} }
	public var nan = NanPropagation.propagate	    { didSet{onSet("nan")} }
	public var reluCeiling = 0.0	                { didSet{onSet("reluCeiling")} }

	//----------------------------------------------------------------------------
	// addAccessors
	public override func addAccessors() {
		super.addAccessors()
		addAccessor(name: "mode",
		            get: { [unowned self] in self.mode },
		            set: { [unowned self] in self.mode = $0 })
		addAccessor(name: "nan",
		            get: { [unowned self] in self.nan },
		            set: { [unowned self] in self.nan = $0 })
		addAccessor(name: "reluCeiling",
		            get: { [unowned self] in self.reluCeiling },
		            set: { [unowned self] in self.reluCeiling = $0 })
	}
}

//==============================================================================
//
public protocol ActivationProperties : ComputableFilterProperties {
	var mode: ActivationMode { get }
	var nan: NanPropagation { get }
	var reluCeiling: Double { get }
}

//--------------------------------------
// ActivationMode
public enum ActivationMode : String, EnumerableType {
	case sigmoid, relu, tanh, clippedRelu
}
