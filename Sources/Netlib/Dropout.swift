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
//  https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf
//
final public class Dropout : ComputableFilterBase, DropoutProperties
{
	//----------------------------------------------------------------------------
	// properties
	public var drop = 0.5                               { didSet{onSet("drop")} }
	public var seed: UInt?                              { didSet{onSet("seed")} }

	//----------------------------------------------------------------------------
	// addAccessors
	public override func addAccessors() {
		super.addAccessors()
		addAccessor(name: "drop",
		            get: { [unowned self] in self.drop },
		            set: { [unowned self] in self.drop = $0 })
		addAccessor(name: "seed" ,
		            get: { [unowned self] in self.seed } ,
		            set: { [unowned self] in self.seed = $0 })
	}

	//----------------------------------------------------------------------------
	// setup
	public override func setup(taskGroup: TaskGroup) throws {
		try super.setup(taskGroup: taskGroup)
		guard 0.0...1.0 ~= drop else {
			writeLog("\(namePath) drop \(drop) is out of range 0.0...1.0")
			throw ModelError.setupFailed
		}
	}
}

//==============================================================================
//
public protocol DropoutProperties : ComputableFilterProperties
{
	var drop: Double { get }
	var seed: UInt? { get }
}
