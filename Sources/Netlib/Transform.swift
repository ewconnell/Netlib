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
import Foundation

//==============================================================================
// Transform
//
public protocol Transform : ModelObject {

}

//==============================================================================
// TransformList
//
public class TransformList : ModelObjectBase, Transform, DefaultPropertiesContainer
{
	//----------------------------------------------------------------------------
	// properties
	public var items = [Transform]()             { didSet{onSet("items")} }

	// DefaultPropertiesContainer
	public var defaults: [Default]? {didSet{onDefaultsChanged(); onSet("defaults")}}
	public var defaultValues: [String : String]? { didSet{onSet("defaultValues")}}
	public var defaultTypeIndex = [String : [String : ModelDefault]]()
	public func onDefaultsChanged() { rebuildDefaultTypeIndex() }

	//----------------------------------------------------------------------------
	// addAccessors
	public override func addAccessors() {
		super.addAccessors()
		addAccessor(name: "defaults",
		            get: { [unowned self] in self.defaults },
		            set: { [unowned self] in self.defaults = $0 })
		addAccessor(name: "defaultValues", lookup: .noLookup,
		            get: { [unowned self] in self.defaultValues },
		            set: { [unowned self] in self.defaultValues = $0 })
		addAccessor(name: "items",
		            get: { [unowned self] in self.items },
		            set: { [unowned self] in self.items = $0 })
	}
}

//==============================================================================
// ScaleTransform
//
public class ScaleTransform : ModelObjectBase, Transform
{
	//----------------------------------------------------------------------------
	// properties
	public var scale = [1.0]	                         { didSet{onSet("scale")} }

	//----------------------------------------------------------------------------
	// addAccessors
	public override func addAccessors() {
		super.addAccessors()
		addAccessor(name: "scale",
		            get: { [unowned self] in self.scale },
		            set: { [unowned self] in self.scale = $0 })
	}
}




