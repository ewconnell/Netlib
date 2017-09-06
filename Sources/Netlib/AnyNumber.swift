//******************************************************************************
//  Created by Edward Connell on 10/11/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
import Foundation

public protocol AnyNumber : BinarySerializable {
	// unchanged cast value
	init(any: AnyNumber)
	init(string: String) throws
	var asUInt8  : UInt8   { get }
	var asUInt16 : UInt16  { get }
	var asInt16  : Int16   { get }
	var asInt32  : Int32   { get }
	var asUInt   : UInt    { get }
	var asInt    : Int     { get }
	var asFloat16: Float16 { get }
	var asFloat  : Float   { get }
	var asDouble : Double  { get }
	var asCVarArg: CVarArg { get }
	var asBool   : Bool    { get }

	// values are normalized to the new type during a cast
	init(norm any: AnyNumber)
	var normUInt8  : UInt8   { get }
	var normUInt16 : UInt16  { get }
	var normInt16  : Int16   { get }
	var normInt32  : Int32   { get }
	var normUInt   : UInt    { get }
	var normInt    : Int     { get }
	var normFloat16: Float16 { get }
	var normFloat  : Float   { get }
	var normDouble : Double  { get }
	var normBool   : Bool    { get }

	var isFiniteValue: Bool { get }
}

//----------------------------------------------------------------------------
extension UInt8 : AnyNumber {
	public init(any: AnyNumber) { self = any.asUInt8 }
	public var asUInt8  : UInt8  { return UInt8(self) }
	public var asUInt16 : UInt16 { return UInt16(self) }
	public var asInt16  : Int16  { return Int16(self) }
	public var asInt32  : Int32  { return Int32(self) }
	public var asUInt   : UInt   { return UInt(self) }
	public var asInt    : Int    { return Int(self) }
	public var asFloat16: Float16{ return Float16(self) }
	public var asFloat  : Float  { return Float(self) }
	public var asDouble : Double { return Double(self) }
	public var asCVarArg: CVarArg{ return self }
	public var asBool   : Bool   { return self != 0 }

	public init(norm any: AnyNumber) { self = any.normUInt8 }
	public static var normScale: Double = 1.0 / (Double(UInt8.max) + 1)
	public static var normScalef: Float = Float(1.0) / (Float(UInt8.max) + 1)
	
	public var normUInt8  : UInt8  { return asUInt8 }
	public var normUInt16 : UInt16 { return asUInt16 }
	public var normInt16  : Int16  { return asInt16 }
	public var normInt32  : Int32  { return asInt32 }
	public var normUInt   : UInt   { return asUInt }
	public var normInt    : Int    { return asInt }
	public var normFloat16: Float16{ return self == 0 ? Float16(0) : Float16((Float(self) + 1) * UInt8.normScalef) }
	public var normFloat  : Float  { return self == 0 ? 0 : (Float(self) + 1) * UInt8.normScalef }
	public var normDouble : Double { return self == 0 ? 0 : (Double(self) + 1) * UInt8.normScale }
	public var normBool   : Bool   { return asBool }

	public var isFiniteValue: Bool { return true }

	public init(string: String) throws {
		guard let value = UInt8(string) else {
			throw PropertiesError.conversionFailed(type: UInt8.self, value: string)
		}
		self = value
	}
}

//----------------------------------------------------------------------------
extension UInt16 : AnyNumber {
	public init(any: AnyNumber) { self = any.asUInt16 }
	public var asUInt8  : UInt8  { return UInt8(self) }
	public var asUInt16 : UInt16 { return UInt16(self) }
	public var asInt16  : Int16  { return Int16(self) }
	public var asInt32  : Int32  { return Int32(self) }
	public var asUInt   : UInt   { return UInt(self) }
	public var asInt    : Int    { return Int(self) }
	public var asFloat16: Float16{ return Float16(self) }
	public var asFloat  : Float  { return Float(self) }
	public var asDouble : Double { return Double(self) }
	public var asCVarArg: CVarArg{ return self }
	public var asBool   : Bool   { return self != 0 }

	public init(norm any: AnyNumber) { self = any.normUInt16 }
	public static var normScale: Double = 1.0 / (Double(UInt16.max) + 1)
	public static var normScalef: Float = Float(1.0) / (Float(UInt16.max) + 1)

	public var normUInt8  : UInt8  { return asUInt8 }
	public var normUInt16 : UInt16 { return asUInt16 }
	public var normInt16  : Int16  { return asInt16 }
	public var normInt32  : Int32  { return asInt32 }
	public var normUInt   : UInt   { return asUInt }
	public var normInt    : Int    { return asInt }
	public var normFloat16: Float16{ return self == 0 ? Float16(0) : Float16((Float(self) + 1) * UInt16.normScalef) }
	public var normFloat  : Float  { return self == 0 ? 0 : (Float(self) + 1) * UInt16.normScalef }
	public var normDouble : Double { return self == 0 ? 0 : (Double(self) + 1) * UInt16.normScale }
	public var normBool   : Bool   { return asBool }

	public var isFiniteValue: Bool { return true }

	public init(string: String) throws {
		guard let value = UInt16(string) else {
			throw PropertiesError.conversionFailed(type: UInt16.self, value: string)
		}
		self = value
	}
}

//----------------------------------------------------------------------------
extension Int16 : AnyNumber {
	public init(any: AnyNumber) { self = any.asInt16 }
	public var asUInt8  : UInt8  { return UInt8(self) }
	public var asUInt16 : UInt16 { return UInt16(self) }
	public var asInt16  : Int16  { return Int16(self) }
	public var asInt32  : Int32  { return Int32(self) }
	public var asUInt   : UInt   { return UInt(self) }
	public var asInt    : Int    { return Int(self) }
	public var asFloat16: Float16{ return Float16(self) }
	public var asFloat  : Float  { return Float(self) }
	public var asDouble : Double { return Double(self) }
	public var asCVarArg: CVarArg{ return self }
	public var asBool   : Bool   { return self != 0 }

	public init(norm any: AnyNumber) { self = any.normInt16 }
	public static var normScale: Double = 1.0 / (Double(Int16.max) + 1)
	public static var normScalef: Float = Float(1.0) / (Float(Int16.max) + 1)
	
	public var normUInt8  : UInt8  { return asUInt8 }
	public var normUInt16 : UInt16 { return asUInt16 }
	public var normInt16  : Int16  { return asInt16 }
	public var normInt32  : Int32  { return asInt32 }
	public var normUInt   : UInt   { return asUInt }
	public var normInt    : Int    { return asInt }
	public var normFloat16: Float16{ return self == 0 ? Float16(0) : Float16((Float(self) + 1) * Int16.normScalef) }
	public var normFloat  : Float  { return self == 0 ? 0 : (Float(self) + 1) * Int16.normScalef }
	public var normDouble : Double { return self == 0 ? 0 : (Double(self) + 1) * Int16.normScale }
	public var normBool   : Bool   { return asBool }

	public var isFiniteValue: Bool { return true }

	public init(string: String) throws {
		guard let value = Int16(string) else {
			throw PropertiesError.conversionFailed(type: Int16.self, value: string)
		}
		self = value
	}
}

//----------------------------------------------------------------------------
extension Int32 : AnyNumber {
	public init(any: AnyNumber) { self = any.asInt32 }
	public var asUInt8  : UInt8  { return UInt8(self) }
	public var asUInt16 : UInt16 { return UInt16(self) }
	public var asInt16  : Int16  { return Int16(self) }
	public var asInt32  : Int32  { return Int32(self) }
	public var asUInt   : UInt   { return UInt(self) }
	public var asInt    : Int    { return Int(self) }
	public var asFloat16: Float16{ return Float16(self) }
	public var asFloat  : Float  { return Float(self) }
	public var asDouble : Double { return Double(self) }
	public var asCVarArg: CVarArg{ return self }
	public var asBool   : Bool   { return self != 0 }

	public init(norm any: AnyNumber) { self = any.normInt32 }
	public static var normScale: Double = 1.0 / (Double(Int32.max) + 1)
	public static var normScalef: Float = Float(1.0) / (Float(Int32.max) + 1)

	public var normUInt8  : UInt8  { return asUInt8 }
	public var normUInt16 : UInt16 { return asUInt16 }
	public var normInt16  : Int16  { return asInt16 }
	public var normInt32  : Int32  { return asInt32 }
	public var normUInt   : UInt   { return asUInt }
	public var normInt    : Int    { return asInt }
	public var normFloat16: Float16{ return self == 0 ? Float16(0) : Float16((Float(self) + 1) * Int32.normScalef) }
	public var normFloat  : Float  { return self == 0 ? 0 : (Float(self) + 1) * Int32.normScalef }
	public var normDouble : Double { return self == 0 ? 0 : (Double(self) + 1) * Int32.normScale }
	public var normBool   : Bool   { return asBool }

	public var isFiniteValue: Bool { return true }

	public init(string: String) throws {
		guard let value = Int32(string) else {
			throw PropertiesError.conversionFailed(type: Int32.self, value: string)
		}
		self = value
	}
}

//----------------------------------------------------------------------------
extension Int : AnyNumber {
	public init(any: AnyNumber) { self = any.asInt }
	public var asUInt8  : UInt8  { return UInt8(self) }
	public var asUInt16 : UInt16 { return UInt16(self) }
	public var asInt16  : Int16  { return Int16(self) }
	public var asInt32  : Int32  { return Int32(self) }
	public var asUInt   : UInt   { return UInt(self) }
	public var asInt    : Int    { return Int(self) }
	public var asFloat16: Float16{ return Float16(self) }
	public var asFloat  : Float  { return Float(self) }
	public var asDouble : Double { return Double(self) }
	public var asCVarArg: CVarArg{ return self }
	public var asBool   : Bool   { return self != 0 }

	public init(norm any: AnyNumber) { self = any.normInt }
	public static var normScale: Double = 1.0 / (Double(Int.max) + 1)
	public static var normScalef: Float = Float(1.0) / (Float(Int.max) + 1)
	
	public var normUInt8  : UInt8  { return asUInt8 }
	public var normUInt16 : UInt16 { return asUInt16 }
	public var normInt16  : Int16  { return asInt16 }
	public var normInt32  : Int32  { return asInt32 }
	public var normUInt   : UInt   { return asUInt }
	public var normInt    : Int    { return asInt }
	public var normFloat16: Float16{ return self == 0 ? Float16(0) : Float16((Float(self) + 1) * Int.normScalef) }
	public var normFloat  : Float  { return self == 0 ? 0 : (Float(self) + 1) * Int.normScalef }
	public var normDouble : Double { return self == 0 ? 0 : (Double(self) + 1) * Int.normScale }
	public var normBool   : Bool   { return asBool }

	public var isFiniteValue: Bool { return true }

	public init(string: String) throws {
		guard let value = Int(string) else {
			throw PropertiesError.conversionFailed(type: Int.self, value: string)
		}
		self = value
	}
}

//----------------------------------------------------------------------------
extension UInt : AnyNumber {
	public init(any: AnyNumber) { self = any.asUInt }
	public var asUInt8  : UInt8  { return UInt8(self) }
	public var asUInt16 : UInt16 { return UInt16(self) }
	public var asInt16  : Int16  { return Int16(self) }
	public var asInt32  : Int32  { return Int32(self) }
	public var asUInt   : UInt   { return UInt(self) }
	public var asInt    : Int    { return Int(self) }
	public var asFloat16: Float16{ return Float16(any: self) }
	public var asFloat  : Float  { return Float(self) }
	public var asDouble : Double { return Double(self) }
	public var asCVarArg: CVarArg{ return self }
	public var asBool   : Bool   { return self != 0 }

	public init(norm any: AnyNumber) { self = any.normUInt }
	public static var normScale: Double = 1.0 / (Double(UInt.max) + 1)
	public static var normScalef: Float = Float(1.0) / (Float(UInt.max) + 1)

	public var normUInt8  : UInt8  { return asUInt8 }
	public var normUInt16 : UInt16 { return asUInt16 }
	public var normInt16  : Int16  { return asInt16 }
	public var normInt32  : Int32  { return asInt32 }
	public var normUInt   : UInt   { return asUInt }
	public var normInt    : Int    { return asInt }
	public var normFloat16: Float16{ return self == 0 ? Float16(0) : Float16((Float(self) + 1) * UInt.normScalef) }
	public var normFloat  : Float  { return self == 0 ? 0 : (Float(self) + 1) * UInt.normScalef }
	public var normDouble : Double { return self == 0 ? 0 : (Double(self) + 1) * UInt.normScale }
	public var normBool   : Bool   { return asBool }

	public var isFiniteValue: Bool { return true }

	public init(string: String) throws {
		guard let value = UInt(string) else {
			throw PropertiesError.conversionFailed(type: UInt.self, value: string)
		}
		self = value
	}
}

//----------------------------------------------------------------------------
extension Bool : AnyNumber {
	public init(any: AnyNumber) { self = any.asBool }
	public var asUInt8  : UInt8  { return self ? 1 : 0 }
	public var asUInt16 : UInt16 { return self ? 1 : 0 }
	public var asInt16  : Int16  { return self ? 1 : 0 }
	public var asInt32  : Int32  { return self ? 1 : 0 }
	public var asUInt   : UInt   { return self ? 1 : 0 }
	public var asInt    : Int    { return self ? 1 : 0 }
	public var asFloat16: Float16{ return Float16(Float(self ? 1 : 0)) }
	public var asFloat  : Float  { return self ? 1 : 0 }
	public var asDouble : Double { return self ? 1 : 0 }
	public var asCVarArg: CVarArg{ return self.asInt }
	public var asBool   : Bool   { return self }
	public var asString : String { return self ? "true" : "false" }

	public init(norm any: AnyNumber) { self = any.normBool }
	public static var normScale: Double = 1
	public static var normScalef : Float = 1

	public var normUInt8  : UInt8  { return asUInt8 }
	public var normUInt16 : UInt16 { return asUInt16 }
	public var normInt16  : Int16  { return asInt16 }
	public var normInt32  : Int32  { return asInt32 }
	public var normUInt   : UInt   { return asUInt }
	public var normInt    : Int    { return asInt }
	public var normFloat16: Float16{ return Float16(any: Float(any: self) * Bool.normScalef) }
	public var normFloat  : Float  { return Float(any: self) * Bool.normScalef }
	public var normDouble : Double { return Double(any: self) * Bool.normScale}
	public var normBool   : Bool   { return asBool }

	public var isFiniteValue: Bool { return true }

	public init(string: String) throws {
		guard let value = Bool(string) else {
			throw PropertiesError.conversionFailed(type: Bool.self, value: string)
		}
		self = value
	}
}

//----------------------------------------------------------------------------
extension Float16 : AnyNumber {
	public init(any: AnyNumber) { self = any.asFloat16 }
	public var asUInt8  : UInt8  { return UInt8(self) }
	public var asUInt16 : UInt16 { return UInt16(self) }
	public var asInt16  : Int16  { return Int16(self) }
	public var asInt32  : Int32  { return Int32(self) }
	public var asUInt   : UInt   { return UInt(any: self) }
	public var asInt    : Int    { return Int(self) }
	public var asFloat16: Float16{ return Float16(self) }
	public var asFloat  : Float  { return Float(self) }
	public var asDouble : Double { return Double(self) }
	public var asCVarArg: CVarArg{ return asFloat }
	public var asBool   : Bool   { return Float(self) != 0 }

	public init(norm any: AnyNumber) { self = any.normFloat16 }
	
	public var normUInt8  : UInt8  { return UInt8(Float(self)  * Float(UInt8.max))}
	public var normUInt16 : UInt16 { return UInt16(Float(self) * Float(UInt16.max))}
	public var normInt16  : Int16  { return Int16(Float(self)  * Float(Int16.max))}
	public var normInt32  : Int32  { return Int32(Float(self)  * Float(Int32.max))}
	public var normUInt   : UInt   { return UInt(Double(self)  * Double(UInt.max))}
	public var normInt    : Int    { return Int(Double(self)   * Double(Int.max))}
	public var normFloat16: Float16{ return asFloat16 }
	public var normFloat  : Float  { return asFloat }
	public var normDouble : Double { return asDouble }
	public var normBool   : Bool   { return asBool }

	public var isFiniteValue: Bool { return Float(self).isFinite }

	public init(string: String) throws {
		guard let value = Float16(string) else {
			throw PropertiesError.conversionFailed(type: Float16.self, value: string)
		}
		self = value
	}
}

//----------------------------------------------------------------------------
extension Float : AnyNumber {
	public init(any: AnyNumber) { self = any.asFloat }
	public var asUInt8  : UInt8  { return UInt8(self) }
	public var asUInt16 : UInt16 { return UInt16(self) }
	public var asInt16  : Int16  { return Int16(self) }
	public var asInt32  : Int32  { return Int32(self) }
	public var asUInt   : UInt   { return UInt(self) }
	public var asInt    : Int    { return Int(self) }
	public var asFloat16: Float16{ return Float16(self) }
	public var asFloat  : Float  { return Float(self) }
	public var asDouble : Double { return Double(self) }
	public var asCVarArg: CVarArg{ return self }
	public var asBool   : Bool   { return self != 0 }

	public init(norm any: AnyNumber) { self = any.normFloat }
	
	public var normUInt8  : UInt8  { return UInt8(Float(self)  * Float(UInt8.max))}
	public var normUInt16 : UInt16 { return UInt16(Float(self) * Float(UInt16.max))}
	public var normInt16  : Int16  { return Int16(Float(self)  * Float(Int16.max))}
	public var normInt32  : Int32  { return Int32(Float(self)  * Float(Int32.max))}
	public var normUInt   : UInt   { return UInt(Double(self)  * Double(UInt.max))}
	public var normInt    : Int    { return Int(Double(self)   * Double(Int.max))}
	public var normFloat16: Float16{ return asFloat16 }
	public var normFloat  : Float  { return asFloat }
	public var normDouble : Double { return asDouble }
	public var normBool   : Bool   { return asBool }

	public var isFiniteValue: Bool { return self.isFinite }

	public init(string: String) throws {
		guard let value = Float(string) else {
			throw PropertiesError.conversionFailed(type: Float.self, value: string)
		}
		self = value
	}
}

//----------------------------------------------------------------------------
extension Double : AnyNumber {
	public init(any: AnyNumber) { self = any.asDouble }
	public var asUInt8  : UInt8  { return UInt8(self) }
	public var asUInt16 : UInt16 { return UInt16(self) }
	public var asInt16  : Int16  { return Int16(self) }
	public var asInt32  : Int32  { return Int32(self) }
	public var asUInt   : UInt   { return UInt(self) }
	public var asInt    : Int    { return Int(self) }
	public var asFloat16: Float16{ return Float16(self) }
	public var asFloat  : Float  { return Float(self) }
	public var asDouble : Double { return Double(self) }
	public var asCVarArg: CVarArg{ return self }
	public var asBool   : Bool   { return self != 0 }

	public init(norm any: AnyNumber) { self = any.normDouble }
	
	public var normUInt8  : UInt8  { return UInt8(Float(self)  * Float(UInt8.max))}
	public var normUInt16 : UInt16 { return UInt16(Float(self) * Float(UInt16.max))}
	public var normInt16  : Int16  { return Int16(Float(self)  * Float(Int16.max))}
	public var normInt32  : Int32  { return Int32(Float(self)  * Float(Int32.max))}
	public var normUInt   : UInt   { return UInt(Double(self)  * Double(UInt.max))}
	public var normInt    : Int    { return Int(Double(self)   * Double(Int.max))}
	public var normFloat16: Float16{ return asFloat16 }
	public var normFloat  : Float  { return asFloat }
	public var normDouble : Double { return asDouble }
	public var normBool   : Bool   { return asBool }

	public var isFiniteValue: Bool { return self.isFinite }

	public init(string: String) throws {
		guard let value = Double(string) else {
			throw PropertiesError.conversionFailed(type: Double.self, value: string)
		}
		self = value
	}
}













