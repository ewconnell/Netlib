//******************************************************************************
//  Created by Edward Connell on 5/13/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
import Foundation

//==============================================================================
// AnyProperty
public protocol AnyProperty : class {
	// properties
	var name: String { get }
	var isDefault: Bool { get set }
	var propertyType: PropertyType { get }
	var version: Int { get set }
	var isCopyable: Bool { get set }
	var isGenerated: Bool { get set }
	var isExplicitlySetup: Bool { get set }
	var isSerialized: Bool { get }
	var isTemplated: Bool { get set }
	var lookup: LookupValue { get set }

	// events
	var changed: PropertyChangedEvent { get }
	
	// functions
	func copy(other: AnyProperty)
	func copyAttributes(from other: AnyProperty)
	func enumerateObjects(handler: (Properties) -> Bool) -> Bool
	func incrementVersion()
	func makeDefault()
	func onSet()
	func selectAny(after modelVersion: Int, include options: SelectAnyOptions) -> Any?
	func setContext()
	func setModel(model: Model)
	func setup(taskGroup: TaskGroup) throws
	func updateAny(with updates: Any) throws
	func updateDefaults()
}

//==============================================================================
// Property
public protocol Property : AnyProperty {
	associatedtype RealizedType
	associatedtype ValueType
	typealias Getter = () -> ValueType
	typealias Setter = (ValueType) -> Void

	// properties
	unowned var owner: Properties { get }
	var name: String { get }
	var value: ValueType { get set }
	var defaultValue: ValueType { get set }
	var version: Int { get set }
	var getter: Getter! { get }
	var setter: Setter! { get }

	// these are defined to allow common code to work with the value
	// always as an optional to avoid nearly duplicate implementations
	func set(_ optional: RealizedType?)
	func setDefault(_ optional: RealizedType?)
	func get() -> RealizedType?
}

//------------------------------------------------------------------------------
// misc
public typealias PropertyChangedEvent = Event<Properties>

public let TypeNameKey = "typename"
public let VersionKey  = "version"
public enum LookupValue { case lookup, noLookup }
public enum PropertyType { case attribute, object, collection }
public typealias AnySet = [String : Any]

public enum PropertiesError : Error {
	case duplicateIdentifier
	case unrecognizedIdentifier
	case typeMismatch
	case missingTypeName
	case conversionFailed(type: Any.Type, value: Any)
}

//==============================================================================
// PropertyBase
open class PropertyBase<RT, VT> {
	// types
	public typealias RealizedType = RT
	public typealias ValueType = VT
	public typealias Getter = () -> ValueType
	public typealias Setter = (ValueType) -> Void

	// initializers
	public init(owner: Properties, name: String, propertyType: PropertyType,
	            lookup: LookupValue, get: @escaping Getter, set: @escaping Setter)
	{
		self.owner = owner
		self.name = name
		self.lookup = lookup
		self.propertyType = propertyType
		defaultValue = get()
		getter = get
		setter = set
	}

	//----------------------------------------------------------------------------
	// properties
	public unowned var owner: Properties
	public let name: String
	public var version = 0

	// this indicates if the property should be serialize, json, xml, etc...
	public var isSerialized: Bool { return !(isGenerated || isTemplated) }
	public var lookup: LookupValue
	public var defaultValue: ValueType
	public let propertyType: PropertyType
	public var getter: Getter!
	public var setter: Setter!

	// events
	public var changed = PropertyChangedEvent()

	// set for properties that are computed during setup and
	// are specific to that object instance, such as connections
	public var isCopyable = true

	// set if value is default
	public var isDefault = true

	// set for properties that have not been defined and
	// are assigned a value during setup, such as element names or inputs
	public var isGenerated = false

	// set for properties that are explicitly setup in the setup() function
	// because they require a specific initialization order. Remaining props
	// will be setup in random order in ModelObject.setup
	public var isExplicitlySetup = false
	
	// set for properties that are assigned a template value during setup
	public var isTemplated = false

	public func copyAttributes(from other: AnyProperty) {
		isCopyable  = other.isCopyable
		isDefault   = other.isDefault
		isGenerated = other.isGenerated
		isTemplated = other.isTemplated
	}
	
	public func enumerateObjects(handler: (Properties) -> Bool) -> Bool { return true }
	public func setup(taskGroup: TaskGroup) throws { }
	public func setModel(model: Model) { }
	public func updateDefaults() { fatalError() }
	
	//----------------------------------------------------------------------------
	// onSet
	//	This is called any time the property is set
	public func onSet() {
		isDefault = false
		incrementVersion()
		setContext()
		changed.raise(data: owner)
	}
	
	//----------------------------------------------------------------------------
	// value
	public var value: ValueType {
		get { return getter() }
		set { setter(newValue) }
	}
	
	//----------------------------------------------------------------------------
	// makeDefault
	public func makeDefault() {
		value = defaultValue
		isDefault = true
	}
	
	//----------------------------------------------------------------------------
	// incrementVersion
	public func incrementVersion() {
		version = owner.model?.incrementVersion() ?? version + 1
	}
	
	//----------------------------------------------------------------------------
	// setContext
	public func setContext() {
		guard let model = owner.model else { return }
		updateDefaults()
		if !isDefault { version = model.version }
	}
}

//==============================================================================
// ValuePropertyBase
open class ValuePropertyBase<RT: AnyConvertible, VT> : PropertyBase<RT, VT> {
	public func get() -> RealizedType? { fatalError() }
	public func set(_ optional: RealizedType?) { fatalError() }
	public func getDefault() -> RealizedType? { fatalError() }
	public func setDefault(_ optional: RealizedType?) { fatalError() }

	//----------------------------------------------------------------------------
	// copy
	public func copy(other: AnyProperty) {
		guard isCopyable else { return }
		if other.version > version {
			set((other as! ValuePropertyBase).get())
			copyAttributes(from: other)
			version = other.version
		}
	}
	
	//----------------------------------------------------------------------------
	// updateDefaults
	public override func updateDefaults() {
		guard isDefault && lookup == .lookup else { return }
		
		if let propertyDefault = owner.lookupDefault(property: name) {
			if propertyDefault.object != nil {
				owner.writeLog("\(name): Default ignored." +
					" Object specified where Value is expected")
			} else {
				do {
					try owner.withVersioningDisabled {
						try set(RealizedType(any: propertyDefault.value))
					}
					version = 0
					isDefault = true
					
				} catch {
					owner.writeLog("\(owner.typeName).\(name) default conversion " +
						"failed: \(propertyDefault.value)")
				}
			}
		}
	}
	
	//----------------------------------------------------------------------------
	// selectAny
	public func selectAny(after modelVersion: Int,
	                      include options: SelectAnyOptions) -> Any? {
		guard version > modelVersion else { return nil }
		return (isDefault ? getDefault() : get())?.asAny ?? NSNull()
	}
	
	//----------------------------------------------------------------------------
	// updateAny
	public func updateAny(with value: Any) throws {
		try set(RealizedType(any: value))
	}
}

// simple value
final public class ValueProperty<T: AnyConvertible> :
ValuePropertyBase<T, T>, AnyProperty, Property {
	public override func get() -> RealizedType? { return value }
	public override func set(_ optional: RealizedType?) { value = optional! }
	public override func getDefault() -> RealizedType? { return defaultValue }
	public override func setDefault(_ optional: RealizedType?) { defaultValue = optional! }
}

final public class OptionalValueProperty<T: AnyConvertible> :
ValuePropertyBase<T, T?>, AnyProperty, Property {
	public override func get() -> RealizedType? { return value }
	public override func set(_ optional: RealizedType?) { value = optional }
	public override func getDefault() -> RealizedType? { return defaultValue }
	public override func setDefault(_ optional: RealizedType?) { defaultValue = optional }
}

//==============================================================================
// ValueArrayPropertyBase
//
open class ValueArrayPropertyBase<RT, VT> : PropertyBase<RT, VT> where
  RT: RangeReplaceableCollection,
	RT.Iterator.Element: AnyConvertible {

	//----------------------------------------------------------------------------
	public func set(_ optional: RealizedType?) { fatalError() }
	public func setDefault(_ optional: RealizedType?) { fatalError() }
	public func get() -> RealizedType? { fatalError() }

	// copy
	public func copy(other: AnyProperty) {
		guard isCopyable else { return }
		// try to avoid copying defaults
		if other.version > version {
			set((other as! ValueArrayPropertyBase).get())
			copyAttributes(from: other)
			version = other.version
		}
	}
	
	//----------------------------------------------------------------------------
	// updateDefaults
	public override func updateDefaults() {
		guard isDefault && lookup == .lookup else { return }
		
		if let propertyDefault = owner.lookupDefault(property: name) {
			if propertyDefault.object != nil {
				owner.writeLog("\(name): Default ignored." +
					" Object specified where Value is expected")
			} else {
				do {
					try owner.withVersioningDisabled {
						try set(RealizedType(any: propertyDefault.value))
					}
					version = 0
					isDefault = true
					
				} catch {
					owner.writeLog("\(owner.typeName).\(name) default conversion " +
						"failed: \(propertyDefault.value)")
				}
			}
		}
	}
	
	//----------------------------------------------------------------------------
	// selectAny
	public func selectAny(after modelVersion: Int,
	                      include options: SelectAnyOptions) -> Any? {
		guard version > modelVersion else { return nil }
		return get() ?? NSNull()
	}
	
	//----------------------------------------------------------------------------
	// updateAny
	public func updateAny(with value: Any) throws {
		try set(RealizedType(any: value))
	}
}

// simple value array
final public class ValueArrayProperty<T: AnyConvertible> :
ValueArrayPropertyBase<[T], [T]>, AnyProperty, Property {
	public override func set(_ optional: RealizedType?) { value = optional! }
	public override func setDefault(_ optional: RealizedType?) { defaultValue = optional! }
	public override func get() -> RealizedType? { return value }
}

final public class OptionalValueArrayProperty<T: AnyConvertible> :
ValueArrayPropertyBase<[T], [T]?>, AnyProperty, Property {
	public override func set(_ optional: RealizedType?) { value = optional }
	public override func setDefault(_ optional: RealizedType?) { defaultValue = optional }
	public override func get() -> RealizedType? { return value }
}

//==============================================================================
// ValueDictionaryPropertyBase
//
open class ValueDictionaryPropertyBase<RT, VT> : PropertyBase<RT, VT> where
	RT: Collection & ExpressibleByDictionaryLiteral,
	RT.Key: Hashable & AnyConvertible,
	RT.Value: AnyConvertible {

	//----------------------------------------------------------------------------
	public func set(_ optional: RealizedType?) { fatalError() }
	public func setDefault(_ optional: RealizedType?) { fatalError() }
	public func get() -> RealizedType? { fatalError() }

	//----------------------------------------------------------------------------
	// copy
	public func copy(other: AnyProperty) {
		guard isCopyable else { return }
		// try to avoid copying defaults
		if other.version > version {
			set((other as! ValueDictionaryPropertyBase).get())
			copyAttributes(from: other)
			version = other.version
		}
	}
	
	//----------------------------------------------------------------------------
	// updateDefaults
	public override func updateDefaults() {
		guard isDefault && lookup == .lookup else { return }
		
		if let propertyDefault = owner.lookupDefault(property: name) {
			if propertyDefault.object != nil {
				owner.writeLog("\(name): Default ignored." +
					" Object specified where Value is expected")
			} else {
				do {
					try owner.withVersioningDisabled {
						try set(RealizedType(any: propertyDefault.value))
					}
					version = 0
					isDefault = true
					
				} catch {
					owner.writeLog("\(owner.typeName).\(name) default conversion " +
						"failed: \(propertyDefault.value)")
				}
			}
		}
	}
	
	//----------------------------------------------------------------------------
	// selectAny
	public func selectAny(after modelVersion: Int,
	                      include options: SelectAnyOptions) -> Any? {
		guard version > modelVersion else { return nil }
		return get() ?? NSNull()
	}
	
	//----------------------------------------------------------------------------
	// updateAny
	public func updateAny(with value: Any) throws {
		try set(RealizedType(any: value))
	}
}

// simple value dictionary
final public class ValueDictionaryProperty<K, V> :
	ValueDictionaryPropertyBase<[K:V], [K:V]>, AnyProperty, Property
where K: Hashable & AnyConvertible, V: AnyConvertible {
	public override func set(_ optional: RealizedType?) { value = optional! }
	public override func setDefault(_ optional: RealizedType?) { defaultValue = optional! }
	public override func get() -> RealizedType? { return value }
}

final public class OptionalValueDictionaryProperty<K, V> :
	ValueDictionaryPropertyBase<[K:V], [K:V]?>, AnyProperty, Property
where K: Hashable & AnyConvertible, V: AnyConvertible {
	public override func set(_ optional: RealizedType?) { value = optional }
	public override func setDefault(_ optional: RealizedType?) { defaultValue = optional }
	public override func get() -> RealizedType? { return value }
}

//==============================================================================
// ObjectProperty
//	All this indirection is to get around the lack of protocol self conformance
// hopefully it will be fixed in future releases.
final public class ObjectProperty<T: ModelObject> : ObjectPropertyBase<T, T> {
	public override func get() -> ModelObject? { return value }
	public override func setDefault(_ v: ModelObject?) { defaultValue = v! as! T }
	public override func set(_ v: ModelObject?) { value = v! as! T }
}

final public class OptionalObjectProperty<T: ModelObject> : ObjectPropertyBase<T, T?> {
	public override func get() -> ModelObject? { return value }
	public override func setDefault(_ v: ModelObject?) { defaultValue = v as? T }
	public override func set(_ v: ModelObject?) { value = v as? T }
}

//--------------------------------------
// ModelObjectProperty
final public class ModelObjectProperty : ObjectPropertyBase<ModelObject, ModelObject> {
	public override func get() -> ModelObject? { return value }
	public override func setDefault(_ v: ModelObject?) { defaultValue = v! }
	public override func set(_ v: ModelObject?) { value = v! }
}

final public class OptionalModelObjectProperty : ObjectPropertyBase<ModelObject, ModelObject?> {
	public override func get() -> ModelObject? { return value }
	public override func setDefault(_ v: ModelObject?) { defaultValue = v }
	public override func set(_ v: ModelObject?) { value = v }
}

//--------------------------------------
// CodecProperty
final public class CodecProperty : ObjectPropertyBase<Codec, Codec> {
	public override func get() -> ModelObject? { return value }
	public override func setDefault(_ v: ModelObject?) { defaultValue = v! as! Codec }
	public override func set(_ v: ModelObject?) { value = v!  as! Codec }
}

final public class OptionalCodecProperty : ObjectPropertyBase<Codec, Codec?> {
	public override func get() -> ModelObject? { return value }
	public override func setDefault(_ v: ModelObject?) { defaultValue = v as? Codec }
	public override func set(_ v: ModelObject?) { value = v as? Codec }
}

//--------------------------------------
// DataSourceProperty
final public class DataSourceProperty : ObjectPropertyBase<DataSource, DataSource> {
	public override func get() -> ModelObject? { return value }
	public override func setDefault(_ v: ModelObject?) { defaultValue = v! as! DataSource }
	public override func set(_ v: ModelObject?) { value = v!  as! DataSource }
}

final public class OptionalDataSourceProperty : ObjectPropertyBase<DataSource, DataSource?> {
	public override func get() -> ModelObject? { return value }
	public override func setDefault(_ v: ModelObject?) { defaultValue = v as? DataSource }
	public override func set(_ v: ModelObject?) { value = v as? DataSource }
}

//--------------------------------------
// DbProviderProperty
final public class DbProviderProperty : ObjectPropertyBase<DbProvider, DbProvider> {
	public override func get() -> ModelObject? { return value }
	public override func setDefault(_ v: ModelObject?) { defaultValue = v! as! DbProvider }
	public override func set(_ v: ModelObject?) { value = v!  as! DbProvider }
}

final public class OptionalDbProviderProperty : ObjectPropertyBase<DbProvider, DbProvider?> {
	public override func get() -> ModelObject? { return value }
	public override func setDefault(_ v: ModelObject?) { defaultValue = v as? DbProvider }
	public override func set(_ v: ModelObject?) { value = v as? DbProvider }
}

//--------------------------------------
// TransformProperty
final public class TransformProperty : ObjectPropertyBase<Transform, Transform> {
	public override func get() -> ModelObject? { return value }
	public override func setDefault(_ v: ModelObject?) { defaultValue = v! as! Transform }
	public override func set(_ v: ModelObject?) { value = v!  as! Transform }
}

final public class OptionalTransformProperty : ObjectPropertyBase<Transform, Transform?> {
	public override func get() -> ModelObject? { return value }
	public override func setDefault(_ v: ModelObject?) { defaultValue = v as? Transform }
	public override func set(_ v: ModelObject?) { value = v as? Transform }
}

//==============================================================================
// ObjectPropertyBase
public class ObjectPropertyBase<RT, VT> :
PropertyBase<ModelObject, VT>, AnyProperty, Property
{
	//----------------------------------------------------------------------------
	// get/ abstract set
	public func get() -> ModelObject? { fatalError() }
	public func setDefault(_ v: ModelObject?) {  }
	public func set(_ v: ModelObject?) { fatalError() }

	//----------------------------------------------------------------------------
	// copy
	public func copy(other: AnyProperty) {
		guard isCopyable else { return }
		set((other as! ObjectPropertyBase).get()?.copy(type: RealizedType.self))
		copyAttributes(from: other)
		version = other.version
	}

	//----------------------------------------------------------------------------
	// setModel
	public override func setModel(model: Model) {
		get()?.setModel(model: model)
  }

	//----------------------------------------------------------------------------
	// setContext
	// called when owner object is added to a Function
	public override func setContext() {
		// first see if this property value is replaced by a default,
		// then set context on the final object
		lookupSelf()
		get()?.setContext(parent: owner,
		                  typePath: owner.typePath,
				              propPath: "\(owner.propPath).\(name)",
				              namePath: owner.namePath)

		if !isDefault, let model = owner.model { version = model.version }
	}
	
	//----------------------------------------------------------------------------
	// updateDefaults
	public override func updateDefaults() {
		guard lookup == .lookup else { return }

		// First see if this object is replaced, then update
		lookupSelf()
		get()?.updateDefaults()
	}
	
	//----------------------------------------------------------------------------
	// setup
	public override func setup(taskGroup: TaskGroup) throws {
		if let object = get() {
			if owner.willLog(level: .diagnostic) {
				owner.diagnostic("\(object.namespaceName) setup",
					               categories: .setup, trailing: "=")
			}
			try object.setup(taskGroup: taskGroup)
		}
	}
	
	//----------------------------------------------------------------------------
	// enumerateObjects
	public override func enumerateObjects(handler: (Properties) -> Bool) -> Bool {
		return get()?.enumerateObjects(handler: handler) ?? true
	}
	
	//----------------------------------------------------------------------------
	// lookupSelf
	public func lookupSelf() {
		guard owner.parent != nil else { return }

		// try to lookup default for whole object if there is one
		if lookup == .lookup && isDefault {
			if let propertyDefault = owner.lookupDefault(property: name) {
				if let object = propertyDefault.objectCopy {
					owner.withVersioningDisabled {
						set(object)
					}
					version = 0
					isDefault = true
					
				} else {
					owner.writeLog("\(name): Default ignored. Object expected")
				}
			}
		}
	}
	
	//----------------------------------------------------------------------------
	// selectAny
	public func selectAny(after modelVersion: Int,
	                      include options: SelectAnyOptions) -> Any? {
		guard let object = get() else {
			return isDefault && modelVersion >= 0 ? nil : NSNull()
		}

		// if this object version has changed, then all non-default props
		// need to be selected from the object to create a new one
		let rebuild = version > modelVersion || options.contains(.types)
		let subVersion = rebuild ? min(0, modelVersion) : modelVersion

		// select
		if var selected = object.selectAny(after: subVersion, include: options) {
			if rebuild { selected[TypeNameKey] = object.typeName }
			return selected
			
		}	else {
			return rebuild ? [TypeNameKey : object.typeName] : nil
		}
	}
	
	//----------------------------------------------------------------------------
	// updateAny
	public func updateAny(with updates: Any) throws {
		if updates is NSNull { set(nil); return	}
		
		// object
		guard let props = updates as? AnySet else {
			owner.writeLog("TypeMismatch expected: AnySet found: \(updates)" +
				" in: \(#function) line: \(#line)")
			throw PropertiesError.typeMismatch
		}
		
		if let typeName = props[TypeNameKey] as? String {
			// create new object
			let object = try owner.Create(typeName: typeName)
			guard object is RT else {
				owner.writeLog("TypeMismatch expected: \(RT.self) found: \(object.self)")
				throw PropertiesError.typeMismatch
			}
			try object.updateAny(with: props)
			set(object)

		} else {
			// update existing object
			try get()!.updateAny(with: props)
		}
	}
}

//==============================================================================
// ObjectArrayProperty
//  TODO: remove all the special case overloads when the swift compiler
//  support conforming to a base protocol
//
final public class ObjectArrayProperty<T: ModelObject> :
	ObjectArrayPropertyBase<T, [T]> {
	public override func setDefault(_ v: [ModelObject]?) { defaultValue = v! as! [T] }
	public override func set(_ v: [ModelObject]?) { value = v! as! [T] }
}

final public class OptionalObjectArrayProperty<T: ModelObject> :
	ObjectArrayPropertyBase<T, [T]?> {
	public override func setDefault(_ v: [ModelObject]?) { defaultValue = v as? [T] }
	public override func set(_ v: [ModelObject]?) { value = v as? [T] }
}

// ModelObjectArrayProperty
final public class ModelObjectArrayProperty : ObjectArrayPropertyBase<ModelObject, [ModelObject]> {
	public override func setDefault(_ v: [ModelObject]?) { defaultValue = v! }
	public override func set(_ v: [ModelObject]?) { value = v! }
}

final public class OptionalModelObjectArrayProperty : ObjectArrayPropertyBase<ModelObject, [ModelObject]?> {
	public override func setDefault(_ v: [ModelObject]?) { defaultValue = v }
	public override func set(_ v: [ModelObject]?) { value = v }
}

// ModelElementArrayProperty
final public class ModelElementArrayProperty : ObjectArrayPropertyBase<ModelElement, [ModelElement]> {
	public override func setDefault(_ v: [ModelObject]?) { defaultValue = v! as! [ModelElement]}
	public override func set(_ v: [ModelObject]?) {
		let temp = v!
		value = temp as! [ModelElement]
	}
}

final public class OptionalModelElementArrayProperty : ObjectArrayPropertyBase<ModelElement, [ModelElement]?> {
	public override func setDefault(_ v: [ModelObject]?) { defaultValue = v as? [ModelElement] }
	public override func set(_ v: [ModelObject]?) { value = v as? [ModelElement] }
}

// ModelTaskArrayProperty
final public class ModelTaskArrayProperty : ObjectArrayPropertyBase<ModelTask, [ModelTask]> {
	public override func setDefault(_ v: [ModelObject]?) { defaultValue = v! as! [ModelTask]}
	public override func set(_ v: [ModelObject]?) { value = v! as! [ModelTask] }
}

final public class OptionalModelTaskArrayProperty : ObjectArrayPropertyBase<ModelTask, [ModelTask]?> {
	public override func setDefault(_ v: [ModelObject]?) { defaultValue = v as? [ModelTask] }
	public override func set(_ v: [ModelObject]?) { value = v as? [ModelTask] }
}

// TransformArrayProperty
final public class TransformArrayProperty : ObjectArrayPropertyBase<Transform, [Transform]> {
	public override func setDefault(_ v: [ModelObject]?) { defaultValue = v! as! [Transform]}
	public override func set(_ v: [ModelObject]?) { value = v! as! [Transform] }
}

final public class OptionalTransformArrayProperty : ObjectArrayPropertyBase<Transform, [Transform]?> {
	public override func setDefault(_ v: [ModelObject]?) { defaultValue = v as? [Transform] }
	public override func set(_ v: [ModelObject]?) { value = v as? [Transform] }
}

//==============================================================================
// ObjectArrayPropertyBase
public class ObjectArrayPropertyBase<ElementType, VT> :
	PropertyBase<[ModelObject], VT>, AnyProperty, Property {
	//----------------------------------------------------------------------------
	// get/ abstract set
	public func get() -> [ModelObject]? { return value as? [ModelObject] }
	public func setDefault(_ v: [ModelObject]?) { fatalError() }
	public func set(_ v: [ModelObject]?) { fatalError() }

	//----------------------------------------------------------------------------
	// copy
	public func copy(other: AnyProperty) {
		guard isCopyable else { return }
		let array = (other as! ObjectArrayPropertyBase)
			.get()?.map { $0.copy(type: ModelObject.self) }
		set(array)
		copyAttributes(from: other)
		version = other.version
	}
	
	//----------------------------------------------------------------------------
	// setModel
	public override func setModel(model: Model) {
		get()?.forEach { $0.setModel(model: model) }
	}
	
	//----------------------------------------------------------------------------
	// setContext
	// called when owner object is added to a Function
	public override func setContext() {
		get()?.forEach {
			$0.setContext(parent: owner, typePath: "", propPath: "",
			              namePath: owner.namePath)
		}
		if !isDefault, let model = owner.model { version = model.version }
	}
	
	//----------------------------------------------------------------------------
	// setup
	public override func setup(taskGroup: TaskGroup) throws {
		try get()?.forEach {
			if owner.willLog(level: .diagnostic) {
				owner.diagnostic("\($0.namespaceName) setup",
					categories: .setup, trailing: "=")
			}
			try $0.setup(taskGroup: taskGroup)
		}
	}
	
	//----------------------------------------------------------------------------
	// updateDefaults
	public override func updateDefaults() {
		get()?.forEach { $0.updateDefaults() }
	}
	
	//----------------------------------------------------------------------------
	// enumerateObjects
	public override func enumerateObjects(handler: (Properties) -> Bool) -> Bool {
		guard let items = get() else { return true }
		
		for item in items {
			if !item.enumerateObjects(handler: handler) { return false }
		}
		return true
	}
	
	//----------------------------------------------------------------------------
	// selectAny
	public func selectAny(after modelVersion: Int,
	                      include options: SelectAnyOptions) -> Any? {
		guard let array = get() else {
			return isDefault && modelVersion >= 0 ? nil : NSNull()
		}

		// rebuild whole array if members have changed. Incremental array
		// changes are not supported now, because arrays are value types
		let rebuild = version > modelVersion || options.contains(.types)
		let subVersion = rebuild ? min(0, modelVersion) : modelVersion

		var selectionIsEmpty = !rebuild
		let result = array.map { (item) -> AnySet in
			if var selected = item.selectAny(after: subVersion, include: options) {
				if rebuild { selected[TypeNameKey] = item.typeName }
				selectionIsEmpty = false
				return selected
			} else {
				return rebuild ? [TypeNameKey : item.typeName] : [:]
			}
		}

		return selectionIsEmpty ? nil : result
	}
	
	//----------------------------------------------------------------------------
	// updateAny
	public func updateAny(with updates: Any) throws {
		if updates is NSNull { set(nil); return	}
		
		guard let propArray = updates as? [AnySet] else {
			owner.writeLog(
				"TypeMismatch expected: AnySet found: \(type(of: updates))" +
				" in: \(#function) line: \(#line)")
			throw PropertiesError.typeMismatch
		}
		
		// if type names are not present then update existing objects
		if !propArray.isEmpty && propArray[0][TypeNameKey] == nil {
			let array = get()!
			assert(propArray.count == array.count)
			
			for (index, item) in propArray.enumerated() where !item.isEmpty {
				try array[index].updateAny(with: item)
			}
		} else {
			// if type names are present, then rebuild array
			var array = RealizedType()
			
			try propArray.forEach {
				guard let typeName = $0[TypeNameKey] as? String else {
					owner.writeLog("item typeName is missing")
					throw PropertiesError.missingTypeName
				}
				let object = try owner.Create(typeName: typeName)
				guard object is ElementType else {
					owner.writeLog(
						"TypeMismatch expected: \(ElementType.self) found: \(object.self)")
					throw PropertiesError.typeMismatch
				}
				try object.updateAny(with: $0)
				array.append(object)
			}
			set(array)
		}
	}
}

//==============================================================================
// ObjectDictionaryProperty
final public class ObjectDictionaryProperty<T: ModelObject> :
	ObjectDictionaryPropertyBase<T, [String : T]> {
	public override func setDefault(_ v: [String : ModelObject]?) { defaultValue = v! as! [String : T] }
	public override func set(_ v: [String : ModelObject]?) { value = v! as! [String : T] }
}

final public class OptionalObjectDictionaryProperty<T: ModelObject> :
	ObjectDictionaryPropertyBase<T, [String : T]?> {
	public override func setDefault(_ v: [String : ModelObject]?) { defaultValue = v as? [String : T] }
	public override func set(_ v: [String : ModelObject]?) { value = v as? [String : T] }
}

//------------------------------------------------------------------------------
// ModelElementDictionaryProperty
final public class ModelElementDictionaryProperty :
	ObjectDictionaryPropertyBase<ModelElement, [String : ModelElement]> {
	public override func setDefault(_ v: [String : ModelObject]?) { defaultValue = v! as! [String : ModelElement] }
	public override func set(_ v: [String : ModelObject]?) { value = v! as! [String : ModelElement] }
}

final public class OptionalModelElementDictionaryProperty :
	ObjectDictionaryPropertyBase<ModelElement, [String : ModelElement]?> {
	public override func setDefault(_ v: [String : ModelObject]?) { defaultValue = v as? [String : ModelElement] }
	public override func set(_ v: [String : ModelObject]?) { value = v as? [String : ModelElement] }
}

//==============================================================================
// ObjectDictionaryPropertyBase
public class ObjectDictionaryPropertyBase<ElementType, VT> :
	PropertyBase<[String : ModelObject], VT>, AnyProperty, Property {
	//----------------------------------------------------------------------------
	// get / abstract set
	public func get() -> [String : ModelObject]? { return value as? [String : ModelObject] }
	public func setDefault(_ v: [String : ModelObject]?) { fatalError() }
	public func set(_ v: [String : ModelObject]?) { fatalError() }

	//----------------------------------------------------------------------------
	// copy
	public func copy(other: AnyProperty) {
		guard isCopyable else { return }
		if let otherDict = (other as! ObjectDictionaryPropertyBase).get() {
			var dict = RealizedType()
			for (key, value) in otherDict {
				dict[key] = (value.copy(type: ModelObject.self))
			}
			set(dict)
		} else {
			set(nil)
		}
		copyAttributes(from: other)
		version = other.version
	}

	//----------------------------------------------------------------------------
	// setModel
	public override func setModel(model: Model) {
		get()?.values.forEach { $0.setModel(model: model) }
	}
	
	//----------------------------------------------------------------------------
	// setContext
	// called when owner object is added to a Function
	public override func setContext() {
		get()?.values.forEach {
			$0.setContext(parent: owner, typePath: "", propPath: "",
			              namePath: owner.namePath)
		}
		if !isDefault, let model = owner.model { version = model.version }
	}

	//----------------------------------------------------------------------------
	// setup
	public override func setup(taskGroup: TaskGroup) throws {
		try get()?.values.forEach {
			if owner.willLog(level: .diagnostic) {
				owner.diagnostic("\($0.namespaceName) setup",
					categories: .setup, trailing: "=")
			}
			try $0.setup(taskGroup: taskGroup)
		}
	}

	//----------------------------------------------------------------------------
	// updateDefaults
	public override func updateDefaults() {
		get()?.values.forEach { $0.updateDefaults() }
	}

	//----------------------------------------------------------------------------
	// enumerateObjects
	public override func enumerateObjects(handler: (Properties) -> Bool) -> Bool {
		guard let items = get()?.values else { return true }

		for item in items {
			if !item.enumerateObjects(handler: handler) { return false }
		}
		return true
	}

	//----------------------------------------------------------------------------
	// selectAny
	public func selectAny(after modelVersion: Int,
	                      include options: SelectAnyOptions) -> Any? {
		guard let dict = get() else {
			return isDefault && modelVersion >= 0 ? nil : NSNull()
		}

		// rebuild if collection members have changed
		// There are no incremental collection changes for now
		let rebuild = version > modelVersion || options.contains(.types)
		let subVersion = rebuild ? min(0, modelVersion) : modelVersion

		var selectionIsEmpty = !rebuild
		var result = [AnySet]()
		for item in dict.values {
			if var selected = item.selectAny(after: subVersion, include: options) {
				if rebuild { selected[TypeNameKey] = item.typeName }
				selectionIsEmpty = false
				result.append(selected)

			} else {
				result.append(rebuild ? [TypeNameKey : item.typeName] : [:])
			}
		}

		return selectionIsEmpty ? nil : result
	}
	
	//----------------------------------------------------------------------------
	// updateAny
	public func updateAny(with updates: Any) throws {
		if updates is NSNull { set(nil); return	}
		
		guard let propArray = updates as? [AnySet] else {
			owner.writeLog("TypeMismatch expected: AnySet found: \(updates)" +
				" in: \(#function) line: \(#line)")
			throw PropertiesError.typeMismatch
		}
		
		// if type names are not present then update existing objects
		if !propArray.isEmpty && propArray[0][TypeNameKey] == nil {
			let dict = get()!
			assert(propArray.count == dict.count)
			
			var dictIndex = dict.startIndex
			for props in propArray where !props.isEmpty {
				// if we are doing an update then each object must have a name
				// property and the name must already be in the dictionary
				try dict.values[dictIndex].updateAny(with: props)
				dictIndex = dict.index(after: dictIndex)
			}
		} else {
			// if type names are present, then rebuild the dictionary
			// each item in the array is a unique key in the dictionary
			var dict = RealizedType()
			
			try propArray.forEach {
				guard let typeName = $0[TypeNameKey] as? String else {
					owner.writeLog("item typeName is missing")
					throw PropertiesError.missingTypeName
				}
				let object = try owner.Create(typeName: typeName)
				guard object is ElementType else {
					owner.writeLog(
						"TypeMismatch expected: \(ElementType.self) found: \(object.self)")
					throw PropertiesError.typeMismatch
				}
				try object.updateAny(with: $0)

				// add object and raise an error if it's a duplicate
				let key = object.namespaceName
				if dict[key] == nil {
					dict[key] = object
				} else {
					owner.writeLog("\(self.owner.typeName):\(self.owner.namespaceName)" +
						" - duplicate element \"\(key)\" ignored")
					throw PropertiesError.duplicateIdentifier
				}
			}
			set(dict)
		}
	}
}

//==============================================================================
// OrderedNamespaceProperty
final public class OrderedNamespaceProperty<T: ModelObject> :
	OrderedNamespacePropertyBase<T, OrderedNamespace<T>> {
	public override func get() -> RealizedType? { return value }
	public override func setDefault(_ v: RealizedType?) { defaultValue = v! }
	public override func set(_ v: RealizedType?) { value = v! }
}

final public class OptionalOrderedNamespaceProperty<T: ModelObject> :
	OrderedNamespacePropertyBase<T, OrderedNamespace<T>?> {
	public override func get() -> RealizedType? { return value }
	public override func setDefault(_ v: RealizedType?) { defaultValue = v }
	public override func set(_ v: RealizedType?) { value = v }
}

//==============================================================================
// OrderedNamespacePropertyBase
//
public class OrderedNamespacePropertyBase<ElementType: ModelObject, VT> :
	PropertyBase<OrderedNamespace<ElementType>, VT>, AnyProperty, Property
{
	public override init(owner: Properties, name: String,
	                     propertyType: PropertyType, lookup: LookupValue,
	                     get: @escaping Getter, set: @escaping Setter) {
		super.init(owner: owner, name: name, propertyType: propertyType,
		           lookup: lookup, get: get, set: set)
		
		changedHandler = self.get()?.changed.addHandler(
			target: self, handler: OrderedNamespacePropertyBase.onCollectionItemChanged)
	}
	
	//----------------------------------------------------------------------------
	// get/ abstract set
	public func get() -> RealizedType? { fatalError() }
	public func setDefault(_ v: RealizedType?) { fatalError() }
	public func set(_ v: RealizedType?) { fatalError() }

	//----------------------------------------------------------------------------
	// onCollectionItemChanged
	private var changedHandler: Disposable?

	private func onCollectionItemChanged(sender: RealizedType) {
		super.onSet()
	}
	
	//----------------------------------------------------------------------------
	// onSet
	//	This is called any time the property is set
	public override func onSet() {
		changedHandler = self.get()?.changed.addHandler(
			target: self, handler: OrderedNamespacePropertyBase.onCollectionItemChanged)
		super.onSet()
	}

	//----------------------------------------------------------------------------
	// copy
	public func copy(other: AnyProperty) {
		guard isCopyable else { return }
		set((other as! OrderedNamespacePropertyBase).get()?.copy())
		copyAttributes(from: other)
		version = other.version
	}

	//----------------------------------------------------------------------------
	// setModel
	public override func setModel(model: Model) {
		get()?.forEach { $0.setModel(model: model) }
	}
	
	//----------------------------------------------------------------------------
	// setContext
	// called when owner object is added to a Function
	public override func setContext() {
		get()?.forEach {
			$0.setContext(parent: owner, typePath: "", propPath: "",
				                namePath: "\(owner.namePath).\(name)")
		}
		if !isDefault, let model = owner.model { version = model.version }
	}

	//----------------------------------------------------------------------------
	// setup
	public override func setup(taskGroup: TaskGroup) throws {
		try get()?.forEach {
			if owner.willLog(level: .diagnostic) {
				owner.diagnostic("\($0.namespaceName) setup",
					categories: .setup, trailing: "=")
			}
			try $0.setup(taskGroup: taskGroup)
		}
	}

	//----------------------------------------------------------------------------
	// updateDefaults
	public override func updateDefaults() {
		get()?.forEach { $0.updateDefaults() }
	}

	//----------------------------------------------------------------------------
	// enumerateObjects
	public override func enumerateObjects(handler: (Properties) -> Bool) -> Bool {
		guard let items = get() else { return true }

		for item in items {
			if !item.enumerateObjects(handler: handler) { return false }
		}
		return true
	}

	//----------------------------------------------------------------------------
	// selectAny
	public func selectAny(after modelVersion: Int,
	                      include options: SelectAnyOptions) -> Any? {
		guard let namespace = get() else {
			return isDefault && modelVersion >= 0 ? nil : NSNull()
		}

		// rebuild whole array if members have changed. Incremental array
		// changes are not supported now, because arrays are value types
		let rebuild = version > modelVersion || options.contains(.types)
		let subVersion = rebuild ? min(0, modelVersion) : modelVersion

		var selectionIsEmpty = !rebuild
		let result = namespace.itemsIndex.sorted(by: { return $0.1 < $1.1 }).map
		{ (entry) -> AnySet in
			let item = namespace.items[entry.value]

			if var selected = item.selectAny(after: subVersion, include: options) {
				if rebuild { selected[TypeNameKey] = item.typeName }
				selectionIsEmpty = false
				return selected
			} else {
				return rebuild ? [TypeNameKey : item.typeName] : [:]
			}
		}

		return selectionIsEmpty ? nil : result
	}

	//----------------------------------------------------------------------------
	// updateAny
	public func updateAny(with updates: Any) throws {
		if updates is NSNull { set(nil); return	}
		
		guard let propArray = updates as? [AnySet] else {
			owner.writeLog(
				"TypeMismatch expected: AnySet found: \(type(of: updates))" +
				" in: \(#function) line: \(#line)")
			throw PropertiesError.typeMismatch
		}
		
		// if type names are not present then update existing objects
		if !propArray.isEmpty && propArray[0][TypeNameKey] == nil {
			let namespace = get()!
			assert(propArray.count == namespace.count)
			
			for (index, propItem) in propArray.enumerated() where !propItem.isEmpty {
				try namespace[index].updateAny(with: propItem)
			}
		} else {
			// if type names are present, then rebuild array
			let namespace = OrderedNamespace<ElementType>()
			
			try propArray.forEach {
				guard let typeName = $0[TypeNameKey] as? String else {
					owner.writeLog("item typeName is missing")
					throw PropertiesError.missingTypeName
				}
				
				let object = try owner.Create(typeName: typeName)
				guard let element = object as? ElementType else {
					owner.writeLog(
						"TypeMismatch expected: \(ElementType.self) found: \(object.self)")
					throw PropertiesError.typeMismatch
				}
				try element.updateAny(with: $0)
				namespace.append(element)
			}
			set(namespace)
		}
	}
}










