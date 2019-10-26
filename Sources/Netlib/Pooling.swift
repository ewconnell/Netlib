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
public final class Pooling : ComputableFilterBase, PoolingProperties, InitHelper {
	//----------------------------------------------------------------------------
	// properties
	public var mode = PoolingMode.max              { didSet{onSet("mode")} }
	public var nan = NanPropagation.propagate			 { didSet{onSet("nan")} }
	public var pad = [0]                           { didSet{onSet("pad")} }
	public var stride = [2]                        { didSet{onSet("stride")} }
	public var windowSize = [2]                    { didSet{onSet("windowSize")} }

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
		addAccessor(name: "pad",
		            get: { [unowned self] in self.pad },
		            set: { [unowned self] in self.pad = $0 })
		addAccessor(name: "stride",
		            get: { [unowned self] in self.stride },
		            set: { [unowned self] in self.stride = $0 })
		addAccessor(name: "windowSize",
		            get: { [unowned self] in self.windowSize },
		            set: { [unowned self] in self.windowSize = $0 })
	}
}

//==============================================================================
//
public protocol PoolingProperties : ComputableFilterProperties {
	var mode: PoolingMode { get }
	var nan: NanPropagation { get }
	var pad: [Int] { get }
	var stride: [Int] { get }
	var windowSize: [Int] { get }
}

//--------------------------------------
// PoolingMode
public enum PoolingMode : String, EnumerableType {
	case averageExcludePadding, averageIncludePadding, max
}

