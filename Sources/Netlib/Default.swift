//******************************************************************************
//  Created by Edward Connell on 5/13/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
import Foundation

//==============================================================================
// ModelDefault
public protocol ModelDefault : ModelObject {
	var object: ModelObject? { get set }
	var objectCopy: ModelObject? { get }
	var property: String { get set }
	var value: String { get set }
}

//==============================================================================
// Default
public final class Default : ModelObjectBase, ModelDefault, InitHelper {
	// initializers
	public required init() {
		super.init()
		// don't lookup any properties on a Default object
		for prop in properties.values { prop.lookup = .noLookup }
	}

	//----------------------------------------------------------------------------
	// properties
	public var property = ""	                       { didSet{onSet("property")} }
	public var value    = ""                         { didSet{onSet("value")} }
	public var object: ModelObject?                  { didSet{onSet("object")} }
	public var objectCopy: ModelObject? { return (object?.copy(type: ModelObject.self))}

	//----------------------------------------------------------------------------
	// addAccessors
	public override func addAccessors() {
		super.addAccessors()
		addAccessor(name: "object",
		            get: { [unowned self] in self.object },
		            set: { [unowned self] in self.object = $0 })
		addAccessor(name: "property",
		            get: { [unowned self] in self.property },
		            set: { [unowned self] in self.property = $0 })
		addAccessor(name: "value",
		            get: { [unowned self] in self.value },
		            set: { [unowned self] in self.value = $0 })
	}
}

//==============================================================================
// DefaultPropertiesContainer
public protocol DefaultPropertiesContainer : Properties {
	// properties
	var defaults: [Default]? { get set }
	var defaultValues: [String : String]? { get set }
	var defaultTypeIndex: [String : [String : ModelDefault]]  { get set }

	// functions
	func lookup(typePath: String, propPath: String) -> ModelDefault?
	func onDefaultsChanged()
	func rebuildDefaultTypeIndex()
}

extension DefaultPropertiesContainer {
	//----------------------------------------------------------------------------
	// onDefaultValuesChanged
	//	This handler rebuilds the defaultTypeIndex
	public func onDefaultValuesChanged(oldValue: [String : String]?) {
		var currentDefaults = defaults ?? []

		// remove old values
		if let oldKeys = oldValue?.keys {
			for key in oldKeys {
				for i in 0..<currentDefaults.count {
					if currentDefaults[i].property == key {
						_ = currentDefaults.remove(at: i)
					}
				}
			}
		}

		// add new ones
		if let newSet = defaultValues {
			for (key, value) in newSet {
				currentDefaults.append(Default { $0.property = key; $0.value = value })
			}
		}
		defaults = currentDefaults
	}

	//----------------------------------------------------------------------------
	// rebuildDefaultTypeIndex
	//	This handler rebuilds the defaultTypeIndex
	public func rebuildDefaultTypeIndex() {
		// clear existing
		defaultTypeIndex.removeAll()

		if let items = defaults {
			for item in items where !item.property.isEmpty {
				// organize by type
				if let dotRange = item.property.range(of: ".") {
					let type = String(item.property[..<dotRange.lowerBound])
					let propPath = String(item.property[dotRange.upperBound...])
					if defaultTypeIndex[type] != nil {
						if defaultTypeIndex[type]![propPath] == nil {
							defaultTypeIndex[type]![propPath] = item
						} else {
							writeLog("\(typeName) default " +
								"property: \"\(propPath)\" duplicate ignored")
						}
					} else {
						defaultTypeIndex[type] = [propPath : item]
					}
				} else {
					writeLog("properties must have '.' prefix")
				}
			}
		}
		updateDefaults()
	}

	//----------------------------------------------------------------------------
	// lookup
	public func lookup(typePath: String, propPath: String) -> ModelDefault? {
		guard defaultTypeIndex.count > 0 else { return nil }
		
		// attempt
		if willLog(level: .diagnostic) {
			diagnostic("trying lookup of type: \(typePath) prop: \(propPath) " +
				"in \(namespaceName).defaults", categories: .tryDefaultsLookup)
		}

		// split the paths
		var result: ModelDefault?
		let blankType = defaultTypeIndex[""]
		let requestType = typePath.components(separatedBy: ".")
		var requestProp = propPath.components(separatedBy: ".")
		assert(requestType.count == requestProp.count - 1)

		for (ti, type) in requestType.enumerated() {
			// get search path
			var searchPath = requestProp[ti + 1]
			for pi in (ti + 2)..<requestProp.count { searchPath += "." + requestProp[pi] }

			func report(item: ModelDefault) {
				// diagnostic reporting
				if willLog(level: .diagnostic) {
					let searchTypeName = self.defaultTypeIndex[type] == nil ? "" : type
					let match = setText("\(searchTypeName).\(searchPath)", color: .yellow)
					let value = setText(item.object == nil ?
						item.value : item.object!.typeName, color: .yellow)

					diagnostic("lookup matched: \(setText("\(type).\(searchPath)", color: .yellow))" +
						" with default property: \(match) value: \(value) in \(namespaceName).defaults",
						categories: .defaultsLookup)
				}
			}

			// first lookup by type
			if let dictType = defaultTypeIndex[type] {
				if let item = dictType[searchPath] {
					result = item;
					report(item: item)
					break
				}
			}

			// if not found, then lookup with blank type
			if let item = blankType?[searchPath] {
				result = item;
				report(item: item)
				break
			}
		}

		return result
	}
}
