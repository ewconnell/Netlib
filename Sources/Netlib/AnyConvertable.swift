//******************************************************************************
//  Created by Edward Connell on 5/11/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
import Foundation
import Dispatch

//==============================================================================
// AnyConvertible
public protocol AnyConvertible {
	init(any: Any) throws
	var asAny: Any { get }
}

extension AnyConvertible {
	public var asAny: Any { return self }
}

//------------------------------------------------------------------------------
extension AnyConvertible where Self: AnyNumber {
	public init(any: Any) throws {
		switch any {
		case is Self: self = any as! Self
		case let value as String: self = try Self.init(string: value)
		default: throw PropertiesError.conversionFailed(type: Self.self, value: any)
		}
	}
}

extension Bool : AnyConvertible { }
extension Int  : AnyConvertible { }
extension UInt : AnyConvertible { }
extension Float : AnyConvertible { }
extension Double : AnyConvertible { }

extension String : AnyConvertible {
	public init(any: Any) throws { self = String(describing: any) }
}

//------------------------------------------------------------------------------
// Array
extension RangeReplaceableCollection where Iterator.Element: AnyConvertible {
	public init(any: Any) throws {
		switch any {
		case is Self: self = any as! Self
		case is String: self = try Self(any as! String)
		default: fatalError()
		}
	}

	public init(_ string: String) throws {
		var array = Self()
		let items = string.components(separatedBy: ",").map { $0.trim() }
		for item in items { try array.append(Iterator.Element(any: item.trim())) }
		self = array
	}
}

//------------------------------------------------------------------------------
// shuffle
extension MutableCollection where Index == Int {
	/// Shuffle the elements of `self` in-place.
	mutating func shuffle() {
		// empty and single-element collections don't shuffle
		if count < 2 { return }

		for i in startIndex..<endIndex - 1 {
			let j = random_uniform(range: endIndex - i) + i
			if i != j {
				self.swapAt(i, j)
			}
		}
	}

	// backward patch
//	#if swift(>=4)
//  #else
	mutating func swapAt(_ i: Int, _ j: Int) {
		let temp = self[i]
		self[i] = self[j]
		self[j] = temp
	}
//	#endif
}

//------------------------------------------------------------------------------
extension ExpressibleByDictionaryLiteral where Key: Hashable & AnyConvertible, Value: AnyConvertible {
	public init(any: Any) throws {
		switch any {
		case is Self: self = any as! Self
		case is String: self = try Dictionary<Key, Value>(any as! String) as! Self
		default: fatalError()
		}
	}

	public init(_ string: String) throws {
		var dict = [Key : Value]()
		let items = string.components(separatedBy: ",")

		for item in items {
			let pair = item.components(separatedBy: ":")
			guard pair.count == 2 else {
				throw PropertiesError.conversionFailed(type: Value.self, value: item)
			}
			try dict[Key(any: pair[0].trim())] = Value(any: pair[1].trim())
		}
		self = dict as! Self
	}

	public var asAny: Any { return self }
}

//------------------------------------------------------------------------------
// DispatchQoS
extension DispatchQoS : AnyConvertible {

	public init(any: Any) throws {
		guard let value = any as? String else {
			throw PropertiesError
				.conversionFailed(type: DispatchQoS.self, value: any)
		}
		
		switch value {
		case "background"     : self = DispatchQoS(qosClass: .background, relativePriority: 0)
		case "utility"        : self = DispatchQoS(qosClass: .utility, relativePriority: 0)
		case "default"        : self = DispatchQoS(qosClass: .`default`, relativePriority: 0)
		case "userInitiated"  : self = DispatchQoS(qosClass: .userInitiated, relativePriority: 0)
		case "userInteractive": self = DispatchQoS(qosClass: .userInteractive, relativePriority: 0)
		case "unspecified"    : self = DispatchQoS(qosClass: .unspecified, relativePriority: 0)
		default: throw PropertiesError
			.conversionFailed(type: LogLevel.self, value: any)
		}
	}
	public init?(string: String) { try? self.init(any: string) }
	public var asAny: Any {	return String(describing: self.qosClass) }
}

//------------------------------------------------------------------------------
// EnumerableType
public protocol EnumerableType :
	RawRepresentable, AnyConvertible, BinarySerializable {}

extension EnumerableType {
	public init(any: Any) throws {
		guard let value = Self(rawValue: any as! RawValue) else {
			throw PropertiesError.conversionFailed(type: Self.self, value: any)
		}
		self = value
	}
	public var asAny: Any { return rawValue }
	
	public init?(_ text: String) {
		try? self.init(any: text)
	}
}

//------------------------------------------------------------------------------
// UnsafeBufferPointer from UnsafeMutableBufferPointer
extension UnsafeBufferPointer {
	public init(_ mutableBuffer: UnsafeMutableBufferPointer<Element>) {
		self.init(start: mutableBuffer.baseAddress, count: mutableBuffer.count)
	}
}

//------------------------------------------------------------------------------
// Array From
extension Array {
	// TODO: loading data into a swift array sucks. It requires copies
	//       and unnecessary initializations. Do something better here!
	public init(data: Data) throws {
		guard data.count % MemoryLayout<Element>.stride == 0 else {
			throw ModelError.error("Data size is not multiple of Element")
		}
		self = data.withUnsafeBytes {
			Array<Element>(UnsafeBufferPointer<Element>(
				start: $0, count: data.count / MemoryLayout<Element>.stride))
		}
	}
	
	public init(contentsOf url: URL) throws {
		try self.init(data: Data(contentsOf: url))
	}
}

//------------------------------------------------------------------------------
//
extension String {
	public init(timeInterval: TimeInterval) {
		let ms = Int(timeInterval.truncatingRemainder(dividingBy: 1.0) * 1000)
		let ti = Int(timeInterval)
		let seconds = ti % 60
		let minutes = (ti / 60) % 60
		let hours = (ti / 3600)
		
		self = String(format: "%0.2d:%0.2d:%0.2d.%0.3d", hours, minutes, seconds, ms)
	}
	
  public func trim() -> String
	{
		return self.trimmingCharacters(in: .whitespacesAndNewlines)
	}
}
