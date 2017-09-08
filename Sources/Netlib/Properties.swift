//******************************************************************************
//  Created by Edward Connell on 5/13/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
import Foundation


//==============================================================================
// Properties
//  Adopted by objects that want dynamic properties
public protocol Properties : Logging {
	// properties
	var isContainer: Bool { get }
	var name: String? { get set }
	var namePath: String { get set }
	var namespaceName: String { get set }
	var properties: [String : AnyProperty] { get set }
	var propPath: String { get set }
	var typeName: String { get }
	var typePath: String { get set }
	var version: Int { get }
	weak var currentLog: Log? { get set }
	weak var model: Model! { get set }
	weak var parent: Properties? { get set }

	// functions
	init()
	func copy(from other: Properties)
	func enumerateObjects(handler: (Properties) -> Bool) -> Bool
	func findStorageLocation() throws -> URL?
	func lookupDefault(property: String) -> ModelDefault?
	func lookupDefault(typePath: String, propPath: String) -> ModelDefault?
	func makeDefault(propertyName: String)
	func onSet(_ propertyName: String)
	func setContext(parent: Properties, typePath: String, propPath: String, namePath: String)
	func setModel(model: Model)
	func selectAny(after modelVersion: Int, include options: SelectAnyOptions) -> AnySet?
	func updateAny(with updates: AnySet?) throws
	func updateDefaults()
}

//------------------------------------------------------------------------------
// Properties
extension Properties {
	//----------------------------------------------------------------------------
	// copy
	public func copy<T>(type: T.Type) -> T {
		let newObject = Swift.type(of: self).init()
		newObject.copy(from: self)
		return newObject as! T
	}

	public func updateAny(with updates: AnySet?) throws {
		try updateAny(with: updates)
	}

	//----------------------------------------------------------------------------
	// withVersioningDisabled
	public func withVersioningDisabled<R>(fn: () throws -> R) rethrows -> R {
		model?.incrementVersioningDisabledCount()
		defer { model?.decrementVersioningDisabledCount() }
		return try fn()
	}
}

//==============================================================================
// InitHelper
public protocol InitHelper : Properties {
	init(fn: (Self) throws -> Void) rethrows
}

extension InitHelper {
	public init(fn: (Self) throws -> Void) rethrows {
		self.init()
		try fn(self)
	}
}

//==============================================================================
// SelectAnyOptions
public struct SelectAnyOptions: OptionSet {
	public init(rawValue: Int) { self.rawValue = rawValue }
	
	// properties
	public let rawValue: Int
	public static let types   = SelectAnyOptions(rawValue: 1 << 0)
	public static let storage = SelectAnyOptions(rawValue: 1 << 1)
}

//==============================================================================
// Copyable
public protocol Copyable {
	func copy() -> Self
}

extension Copyable where Self: Properties {
	//----------------------------------------------------------------------------
	// copy
	public func copy() -> Self {
		let newObject = type(of: self).init()
		newObject.copy(from: self)
		return newObject
	}
}

extension Copyable where Self: ModelDataContainer {
	//----------------------------------------------------------------------------
	// copyTemplate
	//	no model is defined for the template so no attempt is made to
	// increment the model version or lookup defaults. All desired property
	// values are copied from Self
	public func copyTemplate(index: Int, taskGroup: TaskGroup,
	                         initFunc: (Self) throws -> Void) throws -> Self {
		let templatedObject = copy()
		templatedObject.sourceIndex = index
		try initFunc(templatedObject)
		try templatedObject.setup(taskGroup: taskGroup)
		return templatedObject
	}
}

//==============================================================================
// PropertiesBase
//  Adopted by objects that want dynamic properties
open class PropertiesBase : Properties, ObjectTracking {
	// initializer
	public required init() {
		namespaceName = typeName.lowercased()
		namePath      = namespaceName
		typePath      = typeName
		trackingId    = objectTracker.register(object: self)
		addAccessors()

		// copyable but not serializable
		properties["namespaceName"]!.isGenerated = true
	}
	deinit { objectTracker.remove(trackingId: trackingId) }

	//----------------------------------------------------------------------------
	// Properties
	public lazy var typeName: String = { String(describing: type(of: self)) }()
	public weak var currentLog: Log?
	public weak var model: Model!
	public weak var parent: Properties?
	public var properties = [String : AnyProperty]()
	public var namePath = ""
	public var propPath = ""
	public var typePath = ""
	public var isContainer: Bool { return false }
	public var version: Int { return model?.version ?? 0 }

	// name
	public var namespaceName = ""               { didSet{onSet("namespaceName")} }
	public var name: String? {
		didSet {
			namespaceName = name ?? typeName.lowercased()
			onSet("name")
		}
	}

	// ObjectTracking
	public private(set) var trackingId: Int = 0

	// logging
	public var logLevel = LogLevel.error             { didSet{onSet("logLevel")} }
	public var nestingLevel = 0

	//----------------------------------------------------------------------------
	// addAccessors
	public func addAccessors() {
		addAccessor(name: "logLevel", lookup: .noLookup,
		            get: { [unowned self] in self.logLevel },
		            set: { [unowned self] in self.logLevel = $0 })
		addAccessor(name: "name", lookup: .noLookup,
		            get: { [unowned self] in self.name },
		            set: { [unowned self] in self.name = $0 })
		addAccessor(name: "namespaceName", lookup: .noLookup,
		            get: { [unowned self] in self.namespaceName },
		            set: { [unowned self] in self.namespaceName = $0 })
	}

	//----------------------------------------------------------------------------
	// findStorageLocation
	public func findStorageLocation() throws -> URL? {
		return try parent?.findStorageLocation()
	}
	
	//----------------------------------------------------------------------------
	// setModel
	public func setModel(model: Model) {
		self.model = model
		currentLog = model.currentLog
		properties.values.forEach { $0.setModel(model: model) }
	}

	//----------------------------------------------------------------------------
	// copy(from other
	//	The log is copied for error reporting reasons
	//  The parent and model will be set when the object is
  //	connected to a model
	public func copy(from other: Properties) {
		withVersioningDisabled {
			other.properties.forEach {
				properties[$0.key]!.copy(other: $0.value)
			}
		}
	}

	//----------------------------------------------------------------------------
	// enumerateObjects
	//	Enumerates descendants
	@discardableResult
	public func enumerateObjects(handler: (Properties) -> Bool) -> Bool {
		// check self
		if !handler(self) {
			return false
		} else {
			// check down the tree
			for prop in properties.values where prop.propertyType != .attribute {
				if !prop.enumerateObjects(handler: handler) { return false }
			}
			return true
		}
	}

	public func validateTree() {
		let root = self
		enumerateObjects {
			if $0.model == nil {
				print("Error \($0.typeName) model is nil")
				raise(SIGINT)
			}
			if $0.parent == nil && $0 !== root {
				print("Error \($0.typeName) parent is nil")
				raise(SIGINT)
			}
			return true
		}
	}
	
	//----------------------------------------------------------------------------
	// setContext
	//  set the model context for this object and it's dependents
	public func setContext(parent: Properties, typePath: String,
	                       propPath: String, namePath: String) {
		guard let parentModel = parent.model else { return }
		model         = parentModel
		currentLog    = parentModel.log
		self.parent   = parent
		self.propPath = propPath
		self.namePath = "\(namePath).\(namespaceName)"
		self.typePath = typePath.isEmpty ? typeName : "\(typePath).\(typeName)"
		nestingLevel  = parent.nestingLevel + 1

		if willLog(level: .diagnostic) {
			diagnostic("setContext: \(self.namePath)", categories: .context)
			diagnostic("typePath: \(self.typePath)", categories: .context, indent: 1)
			diagnostic("propPath: \(self.propPath)", categories: .context, indent: 1)
			diagnostic("", categories: .context)
		}

		// set the context for each dependent
		for prop in properties.values { prop.setContext()	}
	}

	//----------------------------------------------------------------------------
	// lookupDefault(property:
	//  this is called by this object's properties
	public func lookupDefault(property: String) -> ModelDefault? {
		return lookupDefault(typePath: typePath, propPath: "\(propPath).\(property)")
	}

	//----------------------------------------------------------------------------
	// lookupDefault(typePath:
	//  this is called by a child object and will pass the request to
	// the parent
	public func lookupDefault(typePath: String, propPath: String) -> ModelDefault? {
		return parent?.lookupDefault(typePath: typePath, propPath: propPath)
	}

	//----------------------------------------------------------------------------
	// selectAny
	public func selectAny(after modelVersion: Int = 0,
	                      include options: SelectAnyOptions = []) -> AnySet? {
		// get props
		var selected: AnySet?
		for (key, prop) in properties where prop.isSerialized {
			// special case exclusions
			if !options.contains(.storage) && key == "storage" { continue }
			
			// select the prop
			if let value = prop.selectAny(after: modelVersion, include: options) {
				if selected == nil { selected = [:] }
				selected![key] = value
			}
		}

		if options.contains(.types) { selected?[TypeNameKey] = typeName }
		return selected
	}

	//----------------------------------------------------------------------------
	// updateAny
	public func updateAny(with updates: AnySet?) throws {
		guard let updates = updates else { return }
		
		try withVersioningDisabled {
			// update properties collection first to avoid multiple lookups
			let defaultsKey = "defaults"
			if let value = updates[defaultsKey] {
				if let prop = properties[defaultsKey] {
					try prop.updateAny(with: value)
				} else {
					throw PropertiesError.unrecognizedIdentifier
				}
			}
			
			// do the rest of the props
			for (key, value) in updates where key != TypeNameKey && key != defaultsKey {
				if let prop = properties[key] {
					prop.isGenerated = false
					try prop.updateAny(with: value)
				} else {
					throw PropertiesError.unrecognizedIdentifier
				}
			}
		}
	}

	//----------------------------------------------------------------------------
	// makeDefault
	public func makeDefault(propertyName: String) {
		properties[propertyName]?.makeDefault()
	}

	//----------------------------------------------------------------------------
	// updateDefaults
	public func updateDefaults() {
		for prop in properties.values {
			prop.updateDefaults()
		}
	}

	//----------------------------------------------------------------------------
	// onSet - called whenever a property is set
	public func onSet(_ propertyName: String) {
		assert(properties[propertyName] != nil,
			"addAccessor is missing for property: \(propertyName) ")
		properties[propertyName]!.onSet()
	}

	//============================================================================
	// accessor helpers
	//	Note: 90% of this code will go away once the Swift compiler supports
	//	protocol existentials (conform to self or base). Ugly for now...
	
	// simple value
	public func addAccessor<T: AnyConvertible>(
		name: String, lookup: LookupValue = .lookup,
		get: @escaping () -> T, set: @escaping (T) -> Void) {
		assert(properties[name] == nil, "only add an accessor once")
		properties[name] = ValueProperty(
			owner: self, name: name, propertyType: .attribute, lookup: lookup, get: get, set: set)
	}

	public func addAccessor<T: AnyConvertible>(
		name: String, lookup: LookupValue = .lookup,
		get: @escaping () -> T?, set: @escaping (T?) -> Void) {
		assert(properties[name] == nil, "only add an accessor once")
		properties[name] = OptionalValueProperty(
			owner: self, name: name, propertyType: .attribute, lookup: lookup, get: get, set: set)
	}

	//----------------------------------------------------------------------------
	// value array
	public func addAccessor<T: AnyConvertible>(
		name: String, lookup: LookupValue = .lookup,
		get: @escaping () -> [T], set: @escaping ([T]) -> Void) {
		assert(properties[name] == nil, "only add an accessor once")
		properties[name] = ValueArrayProperty(
			owner: self, name: name, propertyType: .attribute, lookup: lookup, get: get, set: set)
	}

	public func addAccessor<T: AnyConvertible>(
		name: String, lookup: LookupValue = .lookup,
		get: @escaping () -> [T]?, set: @escaping ([T]?) -> Void) {
		assert(properties[name] == nil, "only add an accessor once")
		properties[name] = OptionalValueArrayProperty(
			owner: self, name: name, propertyType: .attribute, lookup: lookup, get: get, set: set)
	}

	//----------------------------------------------------------------------------
	// value dictionary
	public func addAccessor<K: AnyConvertible, V: AnyConvertible>(
		name: String, lookup: LookupValue = .lookup,
		get: @escaping () -> [K:V], set: @escaping ([K:V]) -> Void) {
		assert(properties[name] == nil, "only add an accessor once")
		properties[name] = ValueDictionaryProperty(
			owner: self, name: name, propertyType: .attribute, lookup: lookup, get: get, set: set)
	}

	public func addAccessor<K: AnyConvertible, V: AnyConvertible>(
		name: String, lookup: LookupValue = .lookup,
		get: @escaping () -> [K:V]?, set: @escaping ([K:V]?) -> Void) {
		assert(properties[name] == nil, "only add an accessor once")
		properties[name] = OptionalValueDictionaryProperty(
			owner: self, name: name, propertyType: .attribute, lookup: lookup, get: get, set: set)
	}

	//----------------------------------------------------------------------------
	// concrete ModelObject
	public func addAccessor<T: ModelObject>(
		name: String, lookup: LookupValue = .lookup,
		get: @escaping () -> T, set: @escaping (T) -> Void) {
		assert(properties[name] == nil, "only add an accessor once")
		get().namespaceName = name
		properties[name] = ObjectProperty(
			owner: self, name: name, propertyType: .object, lookup: lookup, get: get, set: set)
	}

	public func addAccessor<T: ModelObject>(
		name: String, lookup: LookupValue = .lookup,
		get: @escaping () -> T?, set: @escaping (T?) -> Void) {
		assert(properties[name] == nil, "only add an accessor once")
		properties[name] = OptionalObjectProperty(
			owner: self, name: name, propertyType: .object, lookup: lookup, get: get, set: set)
	}

	//----------------------------------------------------------------------------
	// object array
	public func addAccessor<T: ModelObject>(
		name: String, lookup: LookupValue = .lookup,
		get: @escaping () -> [T], set: @escaping ([T]) -> Void) {
		assert(properties[name] == nil, "only add an accessor once")
		properties[name] = ObjectArrayProperty(
			owner: self, name: name, propertyType: .collection, lookup: lookup, get: get, set: set)
	}

	public func addAccessor<T: ModelObject>(
		name: String, lookup: LookupValue = .lookup,
		get: @escaping () -> [T]?, set: @escaping ([T]?) -> Void) {
		assert(properties[name] == nil, "only add an accessor once")
		properties[name] = OptionalObjectArrayProperty(
			owner: self, name: name, propertyType: .collection, lookup: lookup, get: get, set: set)
	}

	//----------------------------------------------------------------------------
	// ModelObject
	public func addAccessor(name: String, lookup: LookupValue = .lookup,
	                        get: @escaping () -> ModelObject, set: @escaping (ModelObject) -> Void) {
		assert(properties[name] == nil, "only add an accessor once")
		properties[name] = ModelObjectProperty(
			owner: self, name: name, propertyType: .object, lookup: lookup, get: get, set: set)
	}

	public func addAccessor(name: String, lookup: LookupValue = .lookup,
	                        get: @escaping () -> ModelObject?, set: @escaping (ModelObject?) -> Void) {
		assert(properties[name] == nil, "only add an accessor once")
		properties[name] = OptionalModelObjectProperty(
			owner: self, name: name, propertyType: .object, lookup: lookup, get: get, set: set)
	}

	//----------------------------------------------------------------------------
	// Codec object
	public func addAccessor(name: String, lookup: LookupValue = .lookup,
	                        get: @escaping () -> Codec, set: @escaping (Codec) -> Void) {
		assert(properties[name] == nil, "only add an accessor once")
		properties[name] = CodecProperty(
			owner: self, name: name, propertyType: .object, lookup: lookup, get: get, set: set)
	}

	public func addAccessor(name: String, lookup: LookupValue = .lookup,
	                        get: @escaping () -> Codec?, set: @escaping (Codec?) -> Void) {
		assert(properties[name] == nil, "only add an accessor once")
		properties[name] = OptionalCodecProperty(
			owner: self, name: name, propertyType: .object, lookup: lookup, get: get, set: set)
	}

	//----------------------------------------------------------------------------
	// DataSource object
	public func addAccessor(name: String, lookup: LookupValue = .lookup,
	                        get: @escaping () -> DataSource, set: @escaping (DataSource) -> Void) {
		assert(properties[name] == nil, "only add an accessor once")
		properties[name] = DataSourceProperty(
			owner: self, name: name, propertyType: .object, lookup: lookup, get: get, set: set)
	}

	public func addAccessor(name: String, lookup: LookupValue = .lookup,
	                        get: @escaping () -> DataSource?, set: @escaping (DataSource?) -> Void) {
		assert(properties[name] == nil, "only add an accessor once")
		properties[name] = OptionalDataSourceProperty(
			owner: self, name: name, propertyType: .object, lookup: lookup, get: get, set: set)
	}

	//----------------------------------------------------------------------------
	// DbProvider object
	public func addAccessor(name: String, lookup: LookupValue = .lookup,
	                        get: @escaping () -> DbProvider, set: @escaping (DbProvider) -> Void) {
		assert(properties[name] == nil, "only add an accessor once")
		properties[name] = DbProviderProperty(
			owner: self, name: name, propertyType: .object, lookup: lookup, get: get, set: set)
	}

	public func addAccessor(name: String, lookup: LookupValue = .lookup,
	                        get: @escaping () -> DbProvider?, set: @escaping (DbProvider?) -> Void) {
		assert(properties[name] == nil, "only add an accessor once")
		properties[name] = OptionalDbProviderProperty(
			owner: self, name: name, propertyType: .object, lookup: lookup, get: get, set: set)
	}

	//----------------------------------------------------------------------------
	// Transform object
	public func addAccessor(name: String, lookup: LookupValue = .lookup,
	                        get: @escaping () -> Transform, set: @escaping (Transform) -> Void) {
		assert(properties[name] == nil, "only add an accessor once")
		properties[name] = TransformProperty(
			owner: self, name: name, propertyType: .object, lookup: lookup, get: get, set: set)
	}

	public func addAccessor(name: String, lookup: LookupValue = .lookup,
	                        get: @escaping () -> Transform?, set: @escaping (Transform?) -> Void) {
		assert(properties[name] == nil, "only add an accessor once")
		properties[name] = OptionalTransformProperty(
			owner: self, name: name, propertyType: .object, lookup: lookup, get: get, set: set)
	}

	//----------------------------------------------------------------------------
	// generic OrderedNamespace
	public func addAccessor<T>(name: String, get: @escaping () -> OrderedNamespace<T>,
	                           set: @escaping (OrderedNamespace<T>) -> Void) {
		assert(properties[name] == nil, "only add an accessor once")
		properties[name] = OrderedNamespaceProperty(
			owner: self, name: name, propertyType: .collection, lookup: .noLookup, get: get, set: set)
	}

	public func addAccessor<T>(name: String, get: @escaping () -> OrderedNamespace<T>?,
	                           set: @escaping (OrderedNamespace<T>?) -> Void) {
		assert(properties[name] == nil, "only add an accessor once")
		properties[name] = OptionalOrderedNamespaceProperty(
			owner: self, name: name, propertyType: .collection, lookup: .noLookup, get: get, set: set)
	}

	//----------------------------------------------------------------------------
	// ModelObject array
	public func addAccessor(name: String, get: @escaping () -> [ModelObject],
	                        set: @escaping ([ModelObject]) -> Void) {
		assert(properties[name] == nil, "only add an accessor once")
		properties[name] = ModelObjectArrayProperty(
			owner: self, name: name, propertyType: .collection, lookup: .noLookup, get: get, set: set)
	}

	public func addAccessor(name: String, get: @escaping () -> [ModelObject]?,
	                        set: @escaping ([ModelObject]?) -> Void) {
		assert(properties[name] == nil, "only add an accessor once")
		properties[name] = OptionalModelObjectArrayProperty(
			owner: self, name: name, propertyType: .collection, lookup: .noLookup, get: get, set: set)
	}

	//----------------------------------------------------------------------------
	// ModelElement array
	public func addAccessor(name: String, get: @escaping () -> [ModelElement],
	                        set: @escaping ([ModelElement]) -> Void) {
		assert(properties[name] == nil, "only add an accessor once")
		properties[name] = ModelElementArrayProperty(
			owner: self, name: name, propertyType: .collection, lookup: .noLookup, get: get, set: set)
	}

	public func addAccessor(name: String, get: @escaping () -> [ModelElement]?,
	                        set: @escaping ([ModelElement]?) -> Void) {
		assert(properties[name] == nil, "only add an accessor once")
		properties[name] = OptionalModelElementArrayProperty(
			owner: self, name: name, propertyType: .collection, lookup: .noLookup, get: get, set: set)
	}

	//----------------------------------------------------------------------------
	// ModelTask array
	public func addAccessor(name: String, get: @escaping () -> [ModelTask],
	                        set: @escaping ([ModelTask]) -> Void) {
		assert(properties[name] == nil, "only add an accessor once")
		properties[name] = ModelTaskArrayProperty(
			owner: self, name: name, propertyType: .collection, lookup: .noLookup, get: get, set: set)
	}

	public func addAccessor(name: String, get: @escaping () -> [ModelTask]?,
	                        set: @escaping ([ModelTask]?) -> Void) {
		assert(properties[name] == nil, "only add an accessor once")
		properties[name] = OptionalModelTaskArrayProperty(
			owner: self, name: name, propertyType: .collection, lookup: .noLookup, get: get, set: set)
	}

	//----------------------------------------------------------------------------
	// Transform array
	public func addAccessor(name: String, get: @escaping () -> [Transform],
	                        set: @escaping ([Transform]) -> Void) {
		assert(properties[name] == nil, "only add an accessor once")
		properties[name] = TransformArrayProperty(
			owner: self, name: name, propertyType: .collection, lookup: .noLookup, get: get, set: set)
	}

	public func addAccessor(name: String, get: @escaping () -> [Transform]?,
	                        set: @escaping ([Transform]?) -> Void) {
		assert(properties[name] == nil, "only add an accessor once")
		properties[name] = OptionalTransformArrayProperty(
			owner: self, name: name, propertyType: .collection, lookup: .noLookup, get: get, set: set)
	}

	//----------------------------------------------------------------------------
	// concrete object dictionary
	public func addAccessor<T: ModelObject>(name: String, get: @escaping () -> [String : T],
	                                        set: @escaping ([String : T]) -> Void) {
		assert(properties[name] == nil, "only add an accessor once")
		properties[name] = ObjectDictionaryProperty(
			owner: self, name: name, propertyType: .collection, lookup: .noLookup, get: get, set: set)
	}

	public func addAccessor<T: ModelObject>(name: String, get: @escaping () -> [String : T]?,
	                                        set: @escaping ([String : T]?) -> Void) {
		assert(properties[name] == nil, "only add an accessor once")
		properties[name] = OptionalObjectDictionaryProperty(
			owner: self, name: name, propertyType: .collection, lookup: .noLookup, get: get, set: set)
	}

	//----------------------------------------------------------------------------
	// ModelElement dictionary
	public func addAccessor(name: String, get: @escaping () -> [String : ModelElement],
	                        set: @escaping ([String : ModelElement]) -> Void) {
		assert(properties[name] == nil, "only add an accessor once")
		properties[name] = ModelElementDictionaryProperty(
			owner: self, name: name, propertyType: .collection, lookup: .noLookup, get: get, set: set)
	}

	public func addAccessor(name: String, get: @escaping () -> [String : ModelElement]?,
	                        set: @escaping ([String : ModelElement]?) -> Void) {
		assert(properties[name] == nil, "only add an accessor once")
		properties[name] = OptionalModelElementDictionaryProperty(
			owner: self, name: name, propertyType: .collection, lookup: .noLookup, get: get, set: set)
	}
}
