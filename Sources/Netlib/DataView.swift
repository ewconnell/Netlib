//******************************************************************************
//  Created by Edward Connell on 3/6/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
import Foundation

public struct DataView : BinarySerializable {
	// initializers

	//----------------------------------------------------------------------------
	// fully specified
	public init(shape: Shape,
	            dataType: DataType = .real32F,
	            dataArray: DataArray? = nil,
	            elementOffset: Int = 0,
	            isShared: Bool = false,
	            name: String? = nil,
	            log: Log? = nil) {
		// validate
		assert(dataArray == nil || dataArray!.dataType == dataType)

		self.shape = shape
		self.isShared = isShared
		self.elementOffset = elementOffset
		self.viewByteOffset = elementOffset * dataType.size
		self.viewByteCount = shape.elementSpanCount * dataType.size
		self.log = log

		self.dataArray = dataArray ??
			DataArray(log: log, dataType: dataType,
			          elementCount: shape.elementCount, name: name)
		self.name = name

		assert(viewByteOffset + viewByteCount <= self.dataArray.byteCount)
	}

	//----------------------------------------------------------------------------
	// empty array
	public init() {
		shape = Shape()
		dataArray = DataArray()
		isShared = false
		elementOffset = 0
		viewByteCount = 0
		viewByteOffset = 0
	}
	
	//----------------------------------------------------------------------------
	// simple array
	public init(count: Int, dataType: DataType = .real32F) {
		self.init(shape: Shape(count: count), dataType: dataType)
	}
	
	public init(extent: [Int], dataType: DataType = .real32F) {
		self.init(shape: Shape(extent: extent), dataType: dataType)
	}
	
	//----------------------------------------------------------------------------
	// matrix for BLAS stuff
	public init(rows: Int, cols: Int, dataType: DataType = .real32F) {
		self.init(shape: Shape(rows: rows, cols: cols), dataType: dataType)
	}

	//----------------------------------------------------------------------------
	// BinarySerializable - retained
	public init(from buffer: BufferUInt8, next: inout BufferUInt8) {
		shape = Shape(from: buffer, next: &next)
		dataArray = DataArray(from: buffer, next: &next)
		isShared = false
		elementOffset = 0
		viewByteOffset = 0
		viewByteCount = shape.elementSpanCount * dataArray.dataType.size
		assert(viewByteOffset + viewByteCount <= dataArray.byteCount)
	}
	
	//----------------------------------------------------------------------------
	// copy from pointer
	public init(shape: Shape, dataType: DataType,
	            start: UnsafePointer<UInt8>, count: Int) {
		let buffer = UnsafeBufferPointer(start: start, count: count)
		let dataArray = DataArray(dataType: dataType, buffer: buffer)
		self.init(shape: shape, dataType: dataType, dataArray: dataArray)
	}

	//----------------------------------------------------------------------------
	// copy from array
	public init<T: AnyNumber>(array: [T], shape: Shape) {
		let byteCount = array.count * MemoryLayout<T>.size
		let dataPointer = array.withUnsafeBufferPointer { p in
			p.baseAddress!.withMemoryRebound(to: UInt8.self, capacity: byteCount) { $0 }
		}

		self.init(shape: shape, dataType: DataType(type: T.self),
			        start: dataPointer, count: byteCount)
	}

	public init<T: AnyNumber>(array: [T], shape: Shape? = nil) {
		self.init(array: array, shape: shape ?? Shape(count: array.count))
	}

	//----------------------------------------------------------------------------
	// from other
	//  If self is already in the correct form, then it is returned
	// Otherwise the data is copy/transformed to specification
	public init(from other: DataView,
	            asDataType dataType: DataType? = nil,
	            asShape shape: Shape? = nil,
	            using stream: DeviceStream? = nil,
	            normalizeInts: Bool = false) throws {
		// assign
		let dataType = dataType ?? other.dataType
		let shape = shape ?? other.shape

		assert(shape.elementCount == other.elementCount)
		if dataType == other.dataType && shape == other.shape {
			self = other
		} else {
			var view = DataView(shape: shape, dataType: dataType)
			if let stream = stream {
				try stream.copy(from: other, to: &view, normalizeInts: normalizeInts)
			} else {
				try cpuCopy(from: other, to: &view, normalizeInts: normalizeInts)
			}
			self = view
		}
	}

	//----------------------------------------------------------------------------
	// properties
	public var dataArray: DataArray
	public var dataType : DataType { return dataArray.dataType }

	// shape and shorthand accessors
	public let shape: Shape
	public let elementOffset: Int
	public let viewByteOffset: Int
	public let viewByteCount: Int

	// shorthand access
	public var extent: [Int] { return shape.extent }
	public var strides: [Int] { return shape.strides }
	public var elementCount: Int { return shape.elementCount }
	public var elementSpanCount: Int { return shape.elementSpanCount }
	public var isScalar: Bool { return shape.isScalar }
	public var rank: Int { return shape.rank }
	public var items: Int { return shape.items }
	public var channels: Int { return shape.channels }
	public var rows: Int { return shape.rows }
	public var cols: Int { return shape.cols }
	public var itemStride: Int { return shape.itemStride }
	public var channelStride: Int { return shape.channelStride }
	public var rowStride: Int { return shape.rowStride }
	public var colStride: Int { return shape.colStride }
	public var layout: DataLayout { return shape.layout }

	//----------------------------------------------------------------------------
	// index
	public func index(item: Int, channel: Int, row: Int, col: Int) -> [Int] {
		return shape.index(item: item, channel: channel, row: row, col: col)
	}

	//----------------------------------------------------------------------------
	// makeExtent
	public func makeExtent(items: Int, channels: Int, rows: Int, cols: Int) -> [Int] {
		return shape.makeExtent(items: items, channels: channels, rows: rows, cols: cols)
	}

	//----------------------------------------------------------------------------
	// shared memory
	public var isShared: Bool {
		willSet {
			assert(!newValue || isShared || isUniqueReference(),
				"to set memory to shared it must already be shared or unique")
		}
	}

	// logging
	public weak var log: Log? { didSet { dataArray.currentLog = log } }

	//----------------------------------------------------------------------------
	// name
	private var _name: String?
	public var name: String! {
		get { return _name ?? "DataView" }
		set {
			_name = newValue
			if let newName = newValue { dataArray.name = newName }
		}
	}

	//----------------------------------------------------------------------------
	// for testing
	public private(set) var lastAccessMutated = false

	public mutating func isUniqueReference() -> Bool {
		return isKnownUniquelyReferenced(&dataArray)
	}

	//----------------------------------------------------------------------------
	// helpers to obtain pointers to constants 0 and 1 to support Cuda calls
	public var zero: UnsafeRawPointer { return dataArray.zero.pointer }
	public var one: UnsafeRawPointer { return dataArray.one.pointer }

	//----------------------------------------------------------------------------
	// formatValues
	public func format(columnWidth: Int? = nil,
	                   precision: Int? = nil,
	                   maxItems: Int = Int.max,
	                   maxCols: Int = 10,
	                   highlightThreshold: Float = Float.greatestFiniteMagnitude) -> String {
		// get parameters
		let precision = precision ?? 6
		let columnWidth = columnWidth ?? precision + 3
		let itemCount = min(shape.items, maxItems)
		let colCount = min(shape.cols, maxCols)

		let formats: [DataType : String] = [
			.real8U  : " %3hhu",
			.real16U : " %5hu",
			.real16I : " %5hd",
			.real32I : " %5hd",
			.real16F : "%\(columnWidth).\(precision)f",
			.real32F : "%\(columnWidth).\(precision)f",
			.real64F : "%\(columnWidth).\(precision)f",
		]
		assert(formats[dataType] != nil, "missing entry for data type")
		let format = formats[dataType]!


		// helper
		func printValues<T: AnyNumber>(_ type: T.Type) -> String {
			var string = "DataView extent \(extent.description)\n"
			let normalColor = LogColor.white
			let highlightColor = LogColor.blue
			var currentColor = normalColor
			string += normalColor.rawValue

			// helper to flip highlight color back and forth
			// it doesn't work in the Xcode console for some reason,
			// but it's nice when using CLion
			func setColor(text: inout String, highlight: Bool) {
				#if os(Linux)
				if currentColor == normalColor && highlight {
					text += highlightColor.rawValue
					currentColor = highlightColor

				} else if currentColor == highlightColor && !highlight {
					text += normalColor.rawValue
					currentColor = normalColor
				}
				#endif
			}

			do {
				switch shape.layout {
				//----------------------------------------------------------------------
				case .vector:
					string = "DataView extent [\(shape.items)]\n"
					let pad = itemCount > 9 ? " " : ""

					// get the buffer
					let buffer = try ro(type: T.self)

					for item in 0..<itemCount {
						if item < 10 { string += pad }
						string += "[\(item)] "

						let value = buffer[item * itemStride]
						setColor(text: &string, highlight: value.asFloat > highlightThreshold)
						string += "\(String(format: format, value.asCVarArg))\n"
					}
					string += "\n"

				//----------------------------------------------------------------------
				case .matrix:
					let pad = shape.rows > 9 ? " " : ""

					for row in 0..<itemCount {
						if row < 10 { string += pad }
						string += "[\(row)] "

						// get the row buffer
						let rowView = view(offset: [row, 0], extent: [1, shape.cols])
						let buffer = try rowView.ro(type: T.self)

						for col in 0..<colCount {
							let value = buffer[col * rowView.colStride]
							setColor(text: &string, highlight: value.asFloat > highlightThreshold)
							string += "\(String(format: format, value.asCVarArg))"
						}
						if shape.cols > colCount { string += "..." }
						string += "\n"
					}

				//----------------------------------------------------------------------
				case .nchw:
					let pad = shape.rows > 9 ? " " : ""
					let rowExtent = shape.makeExtent(items: 1, channels: 1, rows: 1, cols: shape.cols)

					for item in 0..<itemCount {
						string += "   item [\(item)] ======================================\n"
						
						for channel in 0..<shape.channels {
							string += "channel [\(channel)] -------------------\n"
							
							for row in 0..<shape.rows {
								if row < 10 { string += pad }
								string += "[\(row)] "
								
								// get the row buffer
								let rowView = view(
									offset: shape.index(item: item, channel: channel, row: row, col: 0),
									extent: rowExtent)
								let buffer = try rowView.ro(type: T.self)

								for col in 0..<colCount {
									let value = buffer[col * rowView.colStride]
									setColor(text: &string, highlight: value.asFloat > highlightThreshold)
									string += "\(String(format: format, value.asCVarArg))"
								}
								if shape.cols > colCount { string += "..." }
								string += "\n"
							}
						}
						string += "\n"
					}

				//----------------------------------------------------------------------
				case .nchw_vector_c: fatalError("not implemented yet")

				//----------------------------------------------------------------------
				case .nhwc:
					let pad = shape.rows > 9 ? " " : ""
					let subscripts = [
						"\u{2080}", "\u{2081}", "\u{2082}", "\u{2083}", "\u{2084}",
						"\u{2085}", "\u{2086}", "\u{2087}", "\u{2088}", "\u{2089}",
						]
					let rowExtent = shape.makeExtent(items: 1, channels: shape.channels,
						rows: 1, cols: shape.cols)

					for item in 0..<itemCount {
						string += "item [\(item)] ======================================\n"

						for row in 0..<shape.rows {
							// initialize line prefixes
							let firstString = row < 10 ? pad + "[\(row)]\u{2080}": "[\(row)]\u{2080}"
							var rowStrings = [firstString]
							let nextPrefix = String(repeating: " ", count: firstString.count - 1)
							for i in 1..<shape.channels {
								if i < subscripts.count {
									rowStrings.append("\(nextPrefix)\(subscripts[i])")
								} else {
									rowStrings.append("\(nextPrefix) ")
								}
							}
							
							// get the row buffer
							let index = shape.index(item: item, channel: 0, row: row, col: 0)
							let rowView = view(offset: index, extent: rowExtent)
							let buffer = try rowView.ro(type: T.self)
							
							// get the channel values for each col
							var colOffset = 0
							for _ in 0..<colCount {
								var chanOffset = colOffset
								for chan in 0..<shape.channels {
									// get the value
									let value = buffer[chanOffset]
									
									// add it to the associate line
									rowStrings[chan] += "\(String(format: format, value.asCVarArg))"
									
									// advance
									chanOffset += rowView.shape.channelStride
								}
								colOffset += rowView.shape.colStride
							}
							
							// terminate channel lines
							for chan in 0..<shape.channels {
								if shape.cols > colCount { rowStrings[0] += "..." }
								string += rowStrings[chan] + "\n"
							}
							if shape.channels > 1 { string += "\n" }
						}
						string += "\n"
					}

//				default: fatalError("not implemented yet")
				}
			} catch {
				string += String(describing: error)
			}
			return string
		}
		
		switch dataType {
		case .real8U:  return printValues(UInt8.self)
		case .real16U: return printValues(UInt16.self)
		case .real16I: return printValues(Int16.self)
		case .real32I: return printValues(Int32.self)
		case .real16F: return printValues(Float16.self)
		case .real32F: return printValues(Float.self)
		case .real64F: return printValues(Double.self)
		}
	}

	//----------------------------------------------------------------------------
	// BinarySerializable
	public func serialize(to buffer: inout [UInt8]) {
		// shape
		if shape.isContiguous {
			shape.serialize(to: &buffer)
		} else {
			// pack into contiguous array for storage
			fatalError("not implemented")
		}
		dataArray.serialize(to: &buffer)
	}

	//----------------------------------------------------------------------------
	// isFinite
	public func isFinite() throws -> Bool {
		var isfiniteValue = true
		func check<T: AnyNumber>(_ type: T.Type) throws {
			let buffer = try ro(type: T.self)
			self.forEachIndex {
				if !buffer[$0].isFiniteValue {
					isfiniteValue = false
				}
			}
		}

		switch dataType {
		case .real16F: try check(Float16.self)
		case .real32F: try check(Float.self)
		case .real64F: try check(Double.self)
		default: isfiniteValue = true
		}
		return isfiniteValue
	}
	
	//----------------------------------------------------------------------------
	// copyIfMutates
	//  Note: this should be called from inside the accessQueue.sync block
	private mutating func copyIfMutates(using stream: DeviceStream? = nil) throws {
		// for unit tests
		lastAccessMutated = false
		guard !isShared && !isUniqueReference() else { return }

		lastAccessMutated = true
		if log?.willLog(level: .diagnostic) == true {
			log!.diagnostic(
				"\(mutationString) \(name ?? "")(\(dataArray.trackingId))  " +
				"elements: \(dataArray.elementCount)",
				categories: [.dataCopy, .dataMutation])
		}

		dataArray = try DataArray(withContentsOf: dataArray, using: stream)
	}

	//----------------------------------------------------------------------------
	// Read only buffer access
	public func roReal8U() throws -> BufferUInt8 {
		// get the queue, if we reference it as a dataArray member it
		// it adds a ref count which messes things up
		let queue = dataArray.accessQueue

		return try queue.sync {
			return try BufferUInt8(
				start: dataArray.roReal8U().baseAddress!.advanced(by: viewByteOffset),
				count: viewByteCount)
		}
	}

	// this version is for accelerator APIs
	public func ro(using stream: DeviceStream) throws -> UnsafeRawPointer {
		// get the queue, if we reference it as a dataArray member it
		// it adds a ref count which messes things up
		let queue = dataArray.accessQueue

		return try queue.sync {
			return try dataArray.ro(using: stream).advanced(by: viewByteOffset)
		}
	}

	public func ro<T: AnyNumber>(type: T.Type) throws -> UnsafeBufferPointer<T> {
		assert(dataType == DataType(type: T.self))
		return try roReal8U().baseAddress!
			.withMemoryRebound(to: T.self, capacity: elementSpanCount) {
			return UnsafeBufferPointer<T>(start: $0, count: elementSpanCount)
		}
	}

	public func ro<T: AnyNumber>(using stream: DeviceStream) throws -> UnsafePointer<T> {
		assert(dataType.size == MemoryLayout<T>.size)
		return try ro(using: stream).bindMemory(to: T.self, capacity: elementSpanCount)
	}
	
	// type specific helpers
	public func roReal16I() throws -> UnsafeBufferPointer<Int16> {
		return try ro(type: Int16.self)
	}
	public func roReal32I() throws -> UnsafeBufferPointer<Int32> {
		return try ro(type: Int32.self)
	}
	public func roReal16U() throws -> UnsafeBufferPointer<UInt16> {
		return try ro(type: UInt16.self)
	}
	public func roReal16F() throws -> UnsafeBufferPointer<Float16> {
		return try ro(type: Float16.self)
	}
	public func roReal32F() throws -> UnsafeBufferPointer<Float> {
		return try ro(type: Float.self)
	}
	public func roReal64F() throws -> UnsafeBufferPointer<Double> {
		return try ro(type: Double.self)
	}
	public func roReal32I(using stream: DeviceStream) throws -> UnsafePointer<Int32> {
		return try ro(using: stream)
	}
	public func roReal16F(using stream: DeviceStream) throws -> UnsafePointer<Float16> {
		return try ro(using: stream)
	}
	public func roReal32F(using stream: DeviceStream) throws -> UnsafePointer<Float> {
		return try ro(using: stream)
	}
	public func roReal64F(using stream: DeviceStream) throws -> UnsafePointer<Double> {
		return try ro(using: stream)
	}

	//----------------------------------------------------------------------------
	// Read Write buffer access
	public mutating func rwReal8U() throws -> MutableBufferUInt8 {
		// get the queue, if we reference it as a dataArray member it
		// it adds a ref count which messes things up
		let queue = dataArray.accessQueue

		return try queue.sync {
			try copyIfMutates()
			return try MutableBufferUInt8(
				start: dataArray.rwReal8U().baseAddress!.advanced(by: viewByteOffset),
				count: viewByteCount)
		}
	}

	// this version is for accelerator APIs
	public mutating func rw(using stream: DeviceStream) throws
			-> UnsafeMutableRawPointer {
		// get the queue, if we reference it as a dataArray member it
		// it adds a ref count which messes things up
		let queue = dataArray.accessQueue

		return try queue.sync {
			try copyIfMutates(using: stream)
			return try dataArray.rw(using: stream).advanced(by: viewByteOffset)
		}
	}

	public mutating func rw<T: AnyNumber>(type: T.Type) throws
			-> UnsafeMutableBufferPointer<T>
	{
		assert(dataType == DataType(type: T.self))
		return try rwReal8U().baseAddress!
			.withMemoryRebound(to: T.self, capacity: elementSpanCount) {
			return UnsafeMutableBufferPointer<T>(start: $0,	count: elementSpanCount)
		}
	}

	public mutating func rw<T: AnyNumber>(using stream: DeviceStream) throws
			-> UnsafeMutablePointer<T> {
		assert(dataType.size == MemoryLayout<T>.size)
		return try rw(using: stream).bindMemory(to: T.self, capacity: elementSpanCount)
	}

	// type specific helpers
	public mutating func rwReal16I() throws -> UnsafeMutableBufferPointer<Int16> {
		return try rw(type: Int16.self)
	}
	public mutating func rwReal32I() throws -> UnsafeMutableBufferPointer<Int32> {
		return try rw(type: Int32.self)
	}
	public mutating func rwReal16U() throws -> UnsafeMutableBufferPointer<UInt16> {
		return try rw(type: UInt16.self)
	}
	public mutating func rwReal16F() throws -> UnsafeMutableBufferPointer<Float16> {
		return try rw(type: Float16.self)
	}
	public mutating func rwReal32F() throws -> UnsafeMutableBufferPointer<Float>  {
		return try rw(type: Float.self)
	}
	public mutating func rwReal64F() throws -> UnsafeMutableBufferPointer<Double> {
		return try rw(type: Double.self)
	}
	public mutating func rwReal16F(using stream: DeviceStream) throws
			-> UnsafeMutablePointer<Float16> {
		return try rw(using: stream)
	}
	public mutating func rwReal32F(using stream: DeviceStream) throws
			-> UnsafeMutablePointer<Float> {
		return try rw(using: stream)
	}
	public mutating func rwReal64F(using stream: DeviceStream) throws
			-> UnsafeMutablePointer<Double> {
		return try rw(using: stream)
	}

	//----------------------------------------------------------------------------
	// get/set at
	public mutating func set<T: AnyNumber>(value: T, at index: [Int]) throws {
		assert(shape.contains(index: index))
		let i = shape.linearIndex(of: index)
		
		switch dataType {
		case .real8U:  return try rwReal8U()[i]  = value.asUInt8
		case .real16U: return try rwReal16U()[i] = value.asUInt16
		case .real16I: return try rwReal16I()[i] = value.asInt16
		case .real32I: return try rwReal32I()[i] = value.asInt32
		case .real16F: return try rwReal16F()[i] = value.asFloat16
		case .real32F: return try rwReal32F()[i] = value.asFloat
		case .real64F: return try rwReal64F()[i] = value.asDouble
		}
	}

	public func get<T: AnyNumber>(at index: [Int]? = nil) throws -> T {
		var i = 0
		if let index = index {
			assert(shape.contains(index: index))
			i = shape.linearIndex(of: index)
		}

		// TODO: a clean combined statement like "return try T(roReal8U()[i])"
		// crashes the compiler, so separate to work around
		switch dataType {
		case .real8U:  let val = try roReal8U()[i];  return T(any: val)
		case .real16U: let val = try roReal16U()[i]; return T(any: val)
		case .real16I: let val = try roReal16I()[i]; return T(any: val)
		case .real32I: let val = try roReal32I()[i]; return T(any: val)
		case .real16F: let val = try roReal16F()[i]; return T(any: val)
		case .real32F: let val = try roReal32F()[i]; return T(any: val)
		case .real64F: let val = try roReal64F()[i]; return T(any: val)
		}
	}

	//----------------------------------------------------------------------------
	// meanAndStd
	public func meanAndStd() throws -> (mean: Double, std: Double) {
		func getStdMean<T: AnyNumber>(_ type: T.Type) throws -> (Double, Double) {
			let array = try ro(type: T.self)
			let count = Double(array.count)
			let mean = array.reduce(0.0, { return $0 + Double(any: $1) }) / count
			let std = sqrt((array.map { pow((Double(any: $0) - mean), 2) }
				.reduce(0.0, +) / count))
			return (mean, std)
		}

		switch dataType {
		case .real8U:  return try getStdMean(UInt8.self)
		case .real16U: return try getStdMean(UInt16.self)
		case .real16I: return try getStdMean(Int16.self)
		case .real32I: return try getStdMean(Int32.self)
		case .real16F: return try getStdMean(Float16.self)
		case .real32F: return try getStdMean(Float.self)
		case .real64F: return try getStdMean(Double.self)
		}
	}

	//----------------------------------------------------------------------------
	// create a sub view
	public func view(offset: [Int], extent: [Int]) -> DataView	{
		// the view created will have the same isShared state as the parent
		return createSubView(offset: offset, extent: extent, isReference: isShared)
	}

	//----------------------------------------------------------------------------
	// view(item:
	public func view(item: Int) -> DataView	{
		return viewItems(offset: item, count: 1)
	}

	//----------------------------------------------------------------------------
	// viewItems
	// the view created will have the same isShared state as the parent
	public func viewItems(offset: Int, count: Int) -> DataView	{
		var index: [Int]
		let viewExtent: [Int]
		if rank == 1 {
			index = [offset]
			viewExtent = [count]
		} else {
			index = [offset] + [Int](repeating: 0, count: rank - 1)
			viewExtent = [count] + shape.extent.suffix(from: 1)
		}

		return createSubView(offset: index, extent: viewExtent, isReference: isShared)
	}

	//----------------------------------------------------------------------------
	// createSubView
	private func createSubView(offset: [Int], extent: [Int], isReference: Bool) -> DataView {
		// validate
		assert(offset.count == shape.rank && extent.count == shape.rank)
		assert(shape.contains(offset: offset,
		                      shape: Shape(extent: extent, layout: shape.layout)))
		assert(extent[0] <= shape.items)

		let eltOffset = elementOffset + shape.linearIndex(of: offset)
		let viewShape = Shape(extent: extent,
			                    layout: shape.layout,
			                    channelFormat: shape.channelFormat,
		                      strides: shape.strides,
			                    colMajor: shape.colMajor)

		return DataView(shape: viewShape, dataType: dataType, dataArray: dataArray,
			              elementOffset: eltOffset, isShared: isReference)
	}

	//----------------------------------------------------------------------------
	// flattened
	public func flattened(axis: Int = 0) -> DataView {
		return createFlattened(axis: axis, isShared: isShared)
	}

	//----------------------------------------------------------------------------
	// createFlattened
	private func createFlattened(axis: Int, isShared: Bool) -> DataView {
		assert(shape.isContiguous, "Cannot reshape strided data")
		assert(axis < shape.rank)
		// check if self already meets requirements
		guard self.isShared != isShared || axis != shape.rank - 1 else {
			return self
		}

		// create a new flat view
		var extent: [Int]
		switch axis {
		case 0: extent = [elementCount]
		case 1: extent = [items, elementCount / items]
		default:
			extent = [Int](shape.extent.prefix(upTo: axis)) +
				[shape.extent.suffix(from: axis).reduce(1, *)] +
				[Int](repeating: 1, count: shape.rank - axis - 1)
		}

		return DataView(shape: Shape(extent: extent), dataType: dataType,
			dataArray: dataArray, elementOffset: elementOffset, isShared: isShared)
	}

	//----------------------------------------------------------------------------
	// reference
	//  creation of a reference is for the purpose of reshaped write
	// operations. Therefore the data will be copied before
	// reference view creation if not uniquely held. References will not
	// be checked on the resulting view when a write pointer is taken
	public mutating func reference(using stream: DeviceStream?) throws -> DataView {
		// get the queue, if we reference it as a dataArray member it
		// it adds a ref count which messes things up
		let queue = dataArray.accessQueue

		return try queue.sync {
			try copyIfMutates(using: stream)
			return DataView(shape: shape, dataType: dataType, dataArray: dataArray,
			                elementOffset: elementOffset, isShared: true)
		}
	}

	//----------------------------------------------------------------------------
	// referenceView
	public mutating func referenceView(offset: [Int], extent: [Int],
	                                   using stream: DeviceStream?) throws -> DataView {
		// get the queue, if we reference it as a dataArray member it
		// it adds a ref count which messes things up
		let queue = dataArray.accessQueue

		return try queue.sync {
			try copyIfMutates(using: stream)
			return createSubView(offset: offset, extent: extent, isReference: true)
		}
	}

	//----------------------------------------------------------------------------
	// referenceFlattened
	public mutating func referenceFlattened(axis: Int = 0,
	                                        using stream: DeviceStream?) throws -> DataView {
		// get the queue, if we reference it as a dataArray member it
		// it adds a ref count which messes things up
		let queue = dataArray.accessQueue

		return try queue.sync {
			try copyIfMutates(using: stream)
			return createFlattened(axis: axis, isShared: true)
		}
	}
}








