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
// Sample
public protocol Sample {
	associatedtype ChannelType: AnyNumber
	
	init(data: DataView) throws
	var buffer: UnsafeBufferPointer<ChannelType> { get set }
	var offset: Int { get set }
	subscript(index: Int) -> ChannelType { get }
}

//==============================================================================
// MutableSample
public protocol MutableSample {
	associatedtype ChannelType: AnyNumber
	
	init(data: inout DataView) throws
	var buffer: UnsafeMutableBufferPointer<ChannelType> { get set }
	var offset: Int { get set }
	subscript(index: Int) -> ChannelType { get set }
}

//==============================================================================
// getChannelOffsets
private func getChannelOffsets(for shape: Shape) -> [Int] {
	var offsets = [Int]()
	for i in 0..<shape.channels {
		offsets.append(i * shape.channelStride)
	}
	return offsets
}

//==============================================================================
// DataSample
public struct DataSample<T: AnyNumber> : Sample {
	public typealias ChannelType = T
	
	// initializer
	public init(data: DataView) throws {
		buffer = try data.ro(type: T.self)
		channelOffset = getChannelOffsets(for: data.shape)
	}
	
	// properties
	public var buffer: UnsafeBufferPointer<ChannelType>
	public var offset: Int = 0
	private let channelOffset: [Int]
	public subscript(index: Int) -> ChannelType {
		return buffer[offset + channelOffset[index]]
	}
}

//==============================================================================
// MutableDataSample
public struct MutableDataSample<T: AnyNumber> : MutableSample {
	public typealias ChannelType = T
	
	// initializer
	public init(data: inout DataView) throws {
		buffer = try data.rw(type: T.self)
		channelOffset = getChannelOffsets(for: data.shape)
	}
	
	// properties
	public var buffer: UnsafeMutableBufferPointer<ChannelType>
	public var offset: Int = 0
	private let channelOffset: [Int]
	public subscript(index: Int) -> ChannelType {
		get { return buffer[offset + channelOffset[index]] }
		set { buffer[offset + channelOffset[index]] = newValue }
	}
}

//==============================================================================
// GraySample
public struct GraySample<T: AnyNumber> : Sample {
	public typealias ChannelType = T

	// initializer
	public init(data: DataView) throws {
		assert(data.channels == 1)
		assert(data.shape.channelFormat == .any || data.shape.channelFormat == .gray)
		buffer = try data.ro(type: T.self)
	}
	
	// properties
	public var buffer: UnsafeBufferPointer<ChannelType>
	public var offset: Int = 0

	// gray
	public var gray: T { return buffer[offset] }
	public subscript(index: Int) -> ChannelType { return buffer[offset] }
}

//==============================================================================
// MutableGraySample
public struct MutableGraySample<T: AnyNumber> : MutableSample {
	public typealias ChannelType = T

	// initializer
	public init(data: inout DataView) throws {
		assert(data.channels == 1)
		assert(data.shape.channelFormat == .any || data.shape.channelFormat == .gray)
		buffer = try data.rw(type: T.self)
	}
	
	// properties
	public var buffer: UnsafeMutableBufferPointer<ChannelType>
	public var offset: Int = 0

	// channels
	public var gray: T {
		get { return buffer[offset] }
		set { buffer[offset] = newValue }
	}
	public subscript(index: Int) -> ChannelType {
		get { return buffer[offset] }
		set { buffer[offset] = newValue }
	}
}

//==============================================================================
// GrayAlphaSample
public struct GrayAlphaSample<T: AnyNumber> : Sample {
	public typealias ChannelType = T
	
	// initializer
	public init(data: DataView) throws {
		assert(data.channels == 2)
		assert(data.shape.channelFormat == .any || data.shape.channelFormat == .grayAlpha)
		buffer = try data.ro(type: T.self)
		channelOffset = getChannelOffsets(for: data.shape)
	}
	
	// properties
	public var buffer: UnsafeBufferPointer<ChannelType>
	public var offset: Int = 0
	private let channelOffset: [Int]
	
	// channels
	public var gray: T { return buffer[offset] }
	public var alpha: T { return buffer[offset + channelOffset[1]] }
	
	public subscript(index: Int) -> ChannelType {
		return buffer[offset + channelOffset[index]]
	}
}

//==============================================================================
// MutableGraySample
public struct MutableGrayAlphaSample<T: AnyNumber> : MutableSample {
	public typealias ChannelType = T
	
	// initializer
	public init(data: inout DataView) throws {
		assert(data.channels == 2)
		assert(data.shape.channelFormat == .any || data.shape.channelFormat == .grayAlpha)
		buffer = try data.rw(type: T.self)
		channelOffset = getChannelOffsets(for: data.shape)
	}
	
	// properties
	public var buffer: UnsafeMutableBufferPointer<ChannelType>
	public var offset: Int = 0
	private let channelOffset: [Int]
	
	// channels
	public var gray: T {
		get { return buffer[offset] }
		set { buffer[offset] = newValue }
	}

	public var alpha: T {
		get { return buffer[offset + channelOffset[1]] }
		set { buffer[offset + channelOffset[1]] = newValue }
	}

	public subscript(index: Int) -> ChannelType {
		get { return buffer[offset + channelOffset[index]] }
		set { buffer[offset + channelOffset[index]] = newValue }
	}
}

//==============================================================================
// RGBSample
public struct RGBSample<T: AnyNumber> : Sample {
	public typealias ChannelType = T
	
	// initializer
	public init(data: DataView) throws {
		assert(data.channels == 3)
		assert(data.shape.channelFormat == .any || data.shape.channelFormat == .rgb)
		self.buffer = try data.ro(type: T.self)
		channelOffset = getChannelOffsets(for: data.shape)
	}
	
	// properties
	public var buffer: UnsafeBufferPointer<ChannelType>
	public var offset: Int = 0
	private let channelOffset: [Int]
	
	// channels
	public var r: T { return buffer[offset + channelOffset[0]] }
	public var g: T { return buffer[offset + channelOffset[1]] }
	public var b: T { return buffer[offset + channelOffset[2]] }
	
	public subscript(index: Int) -> ChannelType {
		get { return buffer[offset + channelOffset[index]] }
	}
}

//==============================================================================
// MutableRGBSample
public struct MutableRGBSample<T: AnyNumber> : MutableSample {
	public typealias ChannelType = T
	
	// initializer
	public init(data: inout DataView) throws {
		assert(data.channels == 3)
		assert(data.shape.channelFormat == .any || data.shape.channelFormat == .rgb)
		self.buffer = try data.rw(type: T.self)
		channelOffset = getChannelOffsets(for: data.shape)
	}
	
	// properties
	public var buffer: UnsafeMutableBufferPointer<ChannelType>
	public var offset: Int = 0
	private let channelOffset: [Int]
	
	// channels
	public var r: T {
		get { return buffer[offset + channelOffset[0]] }
		set { buffer[offset + channelOffset[0]] = newValue }
	}
	
	public var g: T {
		get {return buffer[offset + channelOffset[1]] }
		set { buffer[offset + channelOffset[1]] = newValue }
	}
	
	public var b: T {
		get { return buffer[offset + channelOffset[2]] }
		set { buffer[offset + channelOffset[2]] = newValue }
	}
	
	public subscript(index: Int) -> ChannelType {
		get { return buffer[offset + channelOffset[index]] }
		set { buffer[offset + channelOffset[index]] = newValue }
	}
}

//==============================================================================
// RGBASample
public struct RGBASample<T: AnyNumber> : Sample {
	public typealias ChannelType = T

	// initializer
	public init(data: DataView) throws {
		assert(data.channels == 4)
		assert(data.shape.channelFormat == .any || data.shape.channelFormat == .rgba)
		self.buffer = try data.ro(type: T.self)
		channelOffset = getChannelOffsets(for: data.shape)
	}
	
	// properties
	public var buffer: UnsafeBufferPointer<ChannelType>
	public var offset: Int = 0
	private let channelOffset: [Int]

	// channels
	public var r: T { return buffer[offset + channelOffset[0]] }
	public var g: T { return buffer[offset + channelOffset[1]] }
	public var b: T { return buffer[offset + channelOffset[2]] }
	public var a: T { return buffer[offset + channelOffset[3]] }

	public subscript(index: Int) -> ChannelType {
		get { return buffer[offset + channelOffset[index]] }
	}
}

//==============================================================================
// MutableRGBASample
public struct MutableRGBASample<T: AnyNumber> : MutableSample {
	public typealias ChannelType = T
	
	// initializer
	public init(data: inout DataView) throws {
		assert(data.channels == 4)
		assert(data.shape.channelFormat == .any || data.shape.channelFormat == .rgba)
		self.buffer = try data.rw(type: T.self)
		channelOffset = getChannelOffsets(for: data.shape)
	}
	
	// properties
	public var buffer: UnsafeMutableBufferPointer<ChannelType>
	public var offset: Int = 0
	private let channelOffset: [Int]
	
	// channels
	public var r: T {
		get { return buffer[offset + channelOffset[0]] }
		set { buffer[offset + channelOffset[0]] = newValue }
	}
	
	public var g: T {
		get {return buffer[offset + channelOffset[1]] }
		set { buffer[offset + channelOffset[1]] = newValue }
	}
	
	public var b: T {
		get { return buffer[offset + channelOffset[2]] }
		set { buffer[offset + channelOffset[2]] = newValue }
	}
	
	public var a: T {
		get { return buffer[offset + channelOffset[3]] }
		set { buffer[offset + channelOffset[3]] = newValue }
	}

	public subscript(index: Int) -> ChannelType {
		get { return buffer[offset + channelOffset[index]] }
		set { buffer[offset + channelOffset[index]] = newValue }
	}
}

//==============================================================================
// forEachMutableSample
extension DataView {
	public mutating func forEachMutableSample<T: MutableSample>(
		fn: (inout T) throws -> Void) throws {
		// create sample objects
		var sampleA = try T(data: &self)
		
		// walk 4D tensor samples
		var linearItemA = 0
		for _ in 0..<shape.items {
			var linearRowA = linearItemA
			
			for _ in 0..<shape.rows {
				var linearColA = linearRowA
				
				for _ in 0..<shape.cols {
					sampleA.offset = linearColA
					try fn(&sampleA)
					
					linearColA += shape.colStride
				}
				linearRowA += shape.rowStride
			}
			linearItemA += shape.itemStride
		}
	}

	//==============================================================================
	// forEachMutableSample(other:
	public mutating func forEachMutableSample<T: MutableSample, U: Sample>(
		with other: DataView, fn: (inout T, U) throws -> Void) throws {
		// validate
		assert(shape.items == other.items &&
			shape.rows == other.rows && shape.cols == other.cols)

		// create sample objects
		var sampleA = try T(data: &self)
		var sampleB = try U(data: other)

		// walk 4D tensor samples
		var linearItemA = 0, linearItemB = 0
		for _ in 0..<shape.items {
			var linearRowA = linearItemA
			var linearRowB = linearItemB

			for _ in 0..<shape.rows {
				var linearColA = linearRowA
				var linearColB = linearRowB

				for _ in 0..<shape.cols {
					sampleA.offset = linearColA
					sampleB.offset = linearColB
					try fn(&sampleA, sampleB)

					linearColA += shape.colStride
					linearColB += other.shape.colStride
				}
				linearRowA += shape.rowStride
				linearRowB += other.shape.rowStride
			}
			linearItemA += shape.itemStride
			linearItemB += other.shape.itemStride
		}
	}
}





