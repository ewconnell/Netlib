//******************************************************************************
//  Created by Edward Connell on 5/25/16
//  Copyright © 2016 Connell Research. All rights reserved.
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




