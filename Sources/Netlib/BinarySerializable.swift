//******************************************************************************
//  Created by Edward Connell on 2/24/17
//  Copyright © 2016 Connell Research. All rights reserved.
//
import Foundation

//==============================================================================
// BinarySerializable
public protocol BinarySerializable {
	// to/from buffer
	init(from buffer: BufferUInt8, next: inout BufferUInt8) throws
	func serialize(to buffer: inout [UInt8]) throws

	// to/from stream
	// TODO
}

public typealias BufferUInt8 = UnsafeBufferPointer<UInt8>
public typealias MutableBufferUInt8 = UnsafeMutableBufferPointer<UInt8>

extension UnsafeBufferPointer {
	public init() { self.init(start: nil, count: 0) }
}

//------------------------------------------------------------------------------
// optionals
public func serializeOptional(
	_ optional: BinarySerializable?, to buffer: inout [UInt8]) throws {
	// serialize bool if value is nil or not
	(optional != nil).serialize(to: &buffer)
	try optional?.serialize(to: &buffer)
}

public func deserializeOptional<T: BinarySerializable>(
	from buffer: BufferUInt8, next: inout BufferUInt8) throws -> T? {
	return Bool(from: buffer, next: &next) ?
		try T.init(from: next, next: &next) : nil
}

//------------------------------------------------------------------------------
// simple value serialization
//
extension BinarySerializable {
	public init(from buffer: BufferUInt8, next: inout BufferUInt8) {
		let count = MemoryLayout<Self>.size
		next = BufferUInt8(start: buffer.baseAddress!.advanced(by: count),
			                 count: buffer.count - count)
		self = buffer.baseAddress!.withMemoryRebound(to: Self.self, capacity: 1){$0[0]}
	}

	public func serialize(to buffer: inout [UInt8]) {
		var value = self
		withUnsafeBytes(of: &value) { buffer.append(contentsOf: $0) }
	}
}

//------------------------------------------------------------------------------
// Array
extension Array : BinarySerializable {
	public init(from buffer: BufferUInt8, next: inout BufferUInt8) {
		// load data
		let elementCount = Int(from: buffer, next: &next)
		self = Array(next.baseAddress!
			.withMemoryRebound(to: Element.self, capacity: elementCount) {
			UnsafeBufferPointer(start: $0, count: elementCount)
		})

		// advance
		let byteCount = count * MemoryLayout<Element>.stride
		next = BufferUInt8(start: next.baseAddress!.advanced(by: byteCount),
			                 count: next.count - byteCount)
	}

	public func serialize(to buffer: inout [UInt8]) {
		count.serialize(to: &buffer)
		self.withUnsafeBytes { buffer.append(contentsOf: $0) }
	}
}

//------------------------------------------------------------------------------
// UnsafeBufferPointer
extension UnsafeBufferPointer : BinarySerializable {
	public init(from buffer: BufferUInt8, next: inout BufferUInt8) {
		let count = Int(from: buffer, next: &next)
		let byteCount = count * MemoryLayout<Element>.stride
		self = next.baseAddress!.withMemoryRebound(to: Element.self, capacity: count) {
			UnsafeBufferPointer<Element>(start: $0, count: count)
		}

		next = BufferUInt8(start: next.baseAddress!.advanced(by: byteCount),
			                 count: next.count - byteCount)
	}

	public func serialize(to buffer: inout [UInt8]) {
		let byteCount = count * MemoryLayout<Element>.stride
		count.serialize(to: &buffer)
		baseAddress!.withMemoryRebound(to: UInt8.self, capacity: byteCount) {
			buffer.append(contentsOf: BufferUInt8(start: $0, count: byteCount))
		}
	}
}

//--------------------------------------
// String
extension String : BinarySerializable {
	public init(from buffer: BufferUInt8, next: inout BufferUInt8) {
		self = buffer.baseAddress!.withMemoryRebound(to: CChar.self, capacity: buffer.count){
			String(utf8String: $0)!
		}

		let count = self.lengthOfBytes(using: String.Encoding.utf8)
		next = BufferUInt8(start: buffer.baseAddress!.advanced(by: count + 1),
			                 count: buffer.count - (count + 1))
	}

	public func serialize(to buffer: inout [UInt8]) {
		buffer.append(contentsOf: self.utf8)
		buffer.append(0)
	}
}

//==============================================================================
// TensorHeader
public struct TensorHeader : BinarySerializable {
	public init(dataType: DataType, shape: Shape) {
		self.dataType = dataType
		self.shape = shape
	}

	// properties
	let dataType: DataType
	let shape: Shape

	// *TENSOR* make it 8 bytes for alignment
	static let signature = Int(bigEndian: 0x2A54454E534F522A)

	//----------------------------------------------------------------------------
	// BinarySerializable
	public init(from buffer: BufferUInt8, next: inout BufferUInt8) {
		precondition(isTensor(buffer: buffer, next: &next))
		dataType = DataType(from: next, next: &next)
		shape = Shape(from: next, next: &next)
	}

	public func serialize(to buffer: inout [UInt8]) {
		TensorHeader.signature.serialize(to: &buffer)
		dataType.serialize(to: &buffer)
		shape.serialize(to: &buffer)
	}
}

//------------------------------------------------------------------------------
// isTensor
public func isTensor(buffer: BufferUInt8, next: inout BufferUInt8) -> Bool {
	return buffer.baseAddress!.withMemoryRebound(to: Int.self, capacity: 1) {
		if $0[0] == TensorHeader.signature {
			let count = MemoryLayout<Int>.size
			next = BufferUInt8(start: buffer.baseAddress!.advanced(by: count),
			                   count: buffer.count - (count))
			return true

		} else {
			next = buffer
			return false
		}
	}
}
