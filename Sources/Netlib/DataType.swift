//******************************************************************************
//  Created by Edward Connell on 3/30/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
import ImageCodecs

public enum DataType : Int, AnyConvertible, BinarySerializable {

	case real8U, real16U, real16I, real32I, real16F, real32F, real64F
	
	//----------------------------------------------------------------------------
	// initializers
	// from type
	public init(type: AnyNumber.Type) {
		switch type {
		case is UInt8.Type  : self = .real8U
		case is UInt16.Type : self = .real16U
		case is Int16.Type  : self = .real16I
		case is Int32.Type  : self = .real32I
		case is Float16.Type: self = .real16F
		case is Float.Type  : self = .real32F
		case is Double.Type : self = .real64F
		default: fatalError("Unsupported type")
		}
	}

	public init(type: CDataType) {
		switch type {
		case CDataType_real8U:  self = .real8U
		case CDataType_real16U: self = .real16U
		case CDataType_real16I: self = .real16I
		case CDataType_real32I: self = .real32I
		default: fatalError()
		}
	}

	//----------------------------------------------------------------------------
	// properties
	public var size: Int {
		get {
			switch self {
			case .real16F: return MemoryLayout<Float16>.size
			case .real32F: return MemoryLayout<Float>.size
			case .real64F: return MemoryLayout<Double>.size
			case .real8U : return MemoryLayout<UInt8>.size
			case .real16U: return MemoryLayout<UInt16>.size
			case .real16I: return MemoryLayout<Int16>.size
			case .real32I: return MemoryLayout<Int32>.size
			}
		}
	}

	public var type: AnyNumber.Type {
		get {
			switch self {
			case .real8U : return UInt8.self
			case .real16U: return UInt16.self
			case .real16I: return Int16.self
			case .real32I: return Int32.self
			case .real16F: return Float16.self
			case .real32F: return Float.self
			case .real64F: return Double.self
			}
		}
	}

	public var ctype: CDataType {
		switch self {
		case .real8U:  return CDataType_real8U
		case .real16U: return CDataType_real16U
		case .real16I: return CDataType_real16I
		case .real32I: return CDataType_real32I
		default: fatalError("CDataType \(self) not supported")
		}
	}

	//----------------------------------------------------------------------------
	// AnyConvertible protocol
	public init(any: Any) throws {
		guard let value = any as? String else {
			throw PropertiesError
				.conversionFailed(type: LogLevel.self, value: any)
		}
		
		switch value.lowercased() {
		case "half"   : self = .real16F
		case "real16f": self = .real16F
		case "float"  : self = .real32F
		case "real32f": self = .real32F
		case "double" : self = .real64F
		case "real64f": self = .real64F
		case "real8u" : self = .real8U
		case "uint8"  : self = .real8U
		case "uint16" : self = .real16U
		case "int16"  : self = .real16I
		case "int32"  : self = .real32I
		default:
			throw PropertiesError.conversionFailed(type: LogLevel.self, value: any)
		}
	}
	
	public init?(string: String) { try? self.init(any: string) }

	public var asAny: Any {
		switch self {
		case .real16F: return "real16F"
		case .real32F: return "real32F"
		case .real64F: return "real64F"
		case .real8U : return "real8U"
		case .real16U: return "real16U"
		case .real16I: return "real16I"
		case .real32I: return "real32I"
		}
	}
}
//------------------------------------------------------------------------------
// almostEquals
public func almostEquals<T: AnyNumber>(_ a: T, _ b: T,
                         tolerance: Double = 0.00001) -> Bool {
	return abs(a.asDouble - b.asDouble) < tolerance
}
	
//------------------------------------------------------------------------------
// DataFill
public enum DataFill : String, EnumerableType {
	case constant
	case gaussian
  case indexed
	case msra
	case uniform
	case xavier
	case zero
}

//------------------------------------------------------------------------------
// Memory sizes
extension Int {
	var KB: Int { return self * 1024 }
	var MB: Int { return self * 1024 * 1024 }
	var GB: Int { return self * 1024 * 1024 * 1024 }
	var TB: Int { return self * 1024 * 1024 * 1024 * 1024 }
}

//------------------------------------------------------------------------------
// expand
//	This is for expanding kernel or window sizes
public func expand<T>(array: [T], to rank: Int) -> [T] {
	assert(array.count == 1 || array.count == rank)
	if array.count == 1 {
		return [T](repeating: array[0], count: rank)
	} else {
		return array
	}
}

//------------------------------------------------------------------------------
// NumericType
public protocol NumericType
{
	static func +(lhs: Self, rhs: Self) -> Self
	static func -(lhs: Self, rhs: Self) -> Self
	static func *(lhs: Self, rhs: Self) -> Self
	static func /(lhs: Self, rhs: Self) -> Self
	init(_ v: Self)
	init()
	var size: Int { get }
}

extension NumericType {
	public var size: Int { get { return MemoryLayout<Self>.size } }
}

extension Float16 : NumericType { }
extension Double  : NumericType { }
extension Float   : NumericType { }
extension Int     : NumericType { }
extension Int8    : NumericType { }
extension Int16   : NumericType { }
extension Int32   : NumericType { }
extension Int64   : NumericType { }
extension UInt    : NumericType { }
extension UInt8   : NumericType { }
extension UInt16  : NumericType { }
extension UInt32  : NumericType { }
extension UInt64  : NumericType { }

