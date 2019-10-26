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
final public class LrnCrossChannel :
	ComputableFilterBase, LrnCrossChannelProperties, InitHelper {
	//----------------------------------------------------------------------------
	// properties
	public var alpha = 1e-4                                { didSet{onSet("alpha")} }
	public var beta = 0.75                                 { didSet{onSet("beta")} }
	public var k = 2.0                                     { didSet{onSet("k")} }
	public var mode = LrnCrossChannelMode.crossChannelDim1 { didSet{onSet("mode")} }
	public var windowSize: Int = 5                         { didSet{onSet("windowSize")} }

	//----------------------------------------------------------------------------
	// addAccessors
	public override func addAccessors() {
		super.addAccessors()
		addAccessor(name: "alpha",
		            get: { [unowned self] in self.alpha },
		            set: { [unowned self] in self.alpha = $0 })
		addAccessor(name: "beta",
		            get: { [unowned self] in self.beta },
		            set: { [unowned self] in self.beta = $0 })
		addAccessor(name: "k",
		            get: { [unowned self] in self.k },
		            set: { [unowned self] in self.k = $0 })
		addAccessor(name: "mode",
		            get: { [unowned self] in self.mode },
		            set: { [unowned self] in self.mode = $0 })
		addAccessor(name: "windowSize",
		            get: { [unowned self] in self.windowSize },
		            set: { [unowned self] in self.windowSize = $0 })
	}
}

//==============================================================================
//
public protocol LrnCrossChannelProperties : ComputableFilterProperties {
	var alpha: Double { get }
	var beta: Double { get }
	var k: Double { get }
	var mode: LrnCrossChannelMode { get }
	var windowSize: Int { get }
}

// LrnCrossChannelMode
public enum LrnCrossChannelMode : String, EnumerableType {
	case crossChannelDim1
}



