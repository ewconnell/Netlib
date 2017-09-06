//******************************************************************************
//  Created by Edward Connell on 7/6/17
//  Copyright © 2016 Connell Research. All rights reserved.
//
import Foundation

public struct Shape : Equatable, BinarySerializable {
	// initializers
	public init(extent: [Int],
	            layout: DataLayout? = nil,
	            channelFormat: ChannelFormat = .any,
	            strides: [Int]? = nil,
	            colMajor: Bool = false) {
		// assign
		self.rank = extent.count
		self.extent = extent
		self.colMajor = colMajor

		// layout/channelFormat
		if let dataLayout = layout {
			self.layout = dataLayout
		} else {
			switch extent.count {
			case 1: self.layout = .vector
			case 2: self.layout = .matrix
			case 4: self.layout = .nchw
			default: fatalError("not implemented yet")
			}
		}
		self.channelFormat = channelFormat
		channelsAxis = self.layout.channelsAxis
		rowsAxis = self.layout.rowsAxis
		colsAxis = self.layout.colsAxis
		
		// strides
		if let userStrides = strides {
			self.strides = userStrides
			
		} else if colMajor {
			var cmExtent = extent
			cmExtent.swapAt(rowsAxis, colsAxis)
			var cmStrides = Shape.computeStrides(for: cmExtent)
			cmStrides.swapAt(rowsAxis, colsAxis)
			self.strides = cmStrides

		} else {
			self.strides = Shape.computeStrides(for: extent)
		}
		elementCount = extent.reduce(1, *)
		elementSpanCount = Shape.getSpanCount(extent: extent, strides: self.strides)
	}
	
	//----------------------------------------------------------------------------
	// empty
	public init() {
		colMajor = false
		elementCount = 0
		elementSpanCount = 0
		rank = 0
		extent = []
		strides = []

		// format
		channelFormat = .any
		layout = .vector
		channelsAxis = 1
		rowsAxis = 2
		colsAxis = 3
	}
	
	//----------------------------------------------------------------------------
	// simple array
	public init(count: Int) {
		self.init(extent: [count], layout: .vector)
	}
	
	//----------------------------------------------------------------------------
	// matrix for blas
	public init(rows: Int, cols: Int, colMajor: Bool = false) {
		self.init(extent: [rows, cols], layout: .matrix, colMajor: colMajor)
	}

	//----------------------------------------------------------------------------
	// array of shaped
	public init(items: Int, shape other: Shape, layout: DataLayout? = nil) {
		assert(other.rank >= 1)
		var newLayout: DataLayout
		var newExtent: [Int]

		switch other.rank {
		case 1:
			if other.isScalar {
				assert(layout == nil || layout == .vector)
				newLayout = .vector
				newExtent = [items]
			} else {
				assert(layout == nil || layout == .matrix)
				newLayout = .matrix
				newExtent = [items, other.elementCount]
			}

		case 4:
			newLayout = layout ?? (other.channels == 1 ? .nchw : .nhwc)
			assert(other.items == 1)
			assert(newLayout == .nchw || newLayout == .nhwc)

			if newLayout == .nchw {
				newExtent = [items, other.channels, other.rows, other.cols]
			} else {
				newExtent = [items, other.rows, other.cols, other.channels]
			}

		default: fatalError("not implemented yet")
		}

		// strides must be recalculated if the format changes
		self.init(extent: newExtent,
			        layout: newLayout,
			        channelFormat: other.channelFormat,
		          strides: newLayout == other.layout ? other.strides : nil,
			        colMajor: other.colMajor)
	}

	//----------------------------------------------------------------------------
	// properties
	public let channelFormat: ChannelFormat
	public let colMajor: Bool
	public let elementCount: Int
	public let elementSpanCount: Int
	public let extent: [Int]
	public let layout: DataLayout
	public var isContiguous: Bool { return elementCount == elementSpanCount }
	public var isEmpty: Bool { return elementCount == 0 }
	public let strides: [Int]
	public let channelsAxis: Int
	public let rank: Int
	public let rowsAxis: Int
	public let colsAxis: Int
	
	// access
	public var isScalar: Bool { return rank == 1 && items == 1 }
	public var items: Int { return extent[0] }
	public var channels: Int { return extent[channelsAxis] }
	public var rows: Int { return extent[rowsAxis] }
	public var cols: Int { return extent[colsAxis] }
	public var itemStride: Int { return strides[0] }
	public var channelStride: Int { return strides[channelsAxis] }
	public var rowStride: Int { return strides[rowsAxis] }
	public var colStride: Int { return strides[colsAxis] }
	
	//----------------------------------------------------------------------------
	// linearIndex
	//	returns the linear element index
	public func linearIndex(of index: [Int]) -> Int {
		assert(rank > 0 && index.count == rank)
		var result: Int
		switch rank {
		case 1: result = index[0]
		case 2: result = index[0] * strides[0] + index[1] * strides[1]
		default: result = index[0] * strides[0] + index[1] * strides[1] +
			                index[2] * strides[2] + index[3] * strides[3]
		}
		assert(result <= elementSpanCount)
		return result
	}
	
	//----------------------------------------------------------------------------
	// index helpers
	public func index(item: Int, channel: Int, row: Int, col: Int) -> [Int] {
		assert(layout != .vector && layout != .matrix)
		if layout == .nchw {
			return [item, channel, row, col]
		} else {
			return [item, row, col, channel]
		}
	}

	//----------------------------------------------------------------------------
	// extent helpers
	public func makeExtent(items: Int, channels: Int, rows: Int, cols: Int) -> [Int] {
		assert(rank == 4)
		if layout == .nchw {
			return [items, channels, rows, cols]
		} else {
			return [items, rows, cols, channels]
		}
	}

	//----------------------------------------------------------------------------
	// contains
	public func contains(index: [Int]) -> Bool {
		assert(index.count == extent.count, "rank mismatch")
		return linearIndex(of: index) <= elementSpanCount
	}
	
	public func contains(shape: Shape) -> Bool {
		assert(shape.extent.count == extent.count, "rank mismatch")
		return shape.elementSpanCount <= elementSpanCount
	}
	
	public func contains(offset: [Int], shape: Shape) -> Bool {
		assert(offset.count == extent.count &&
			     shape.extent.count == extent.count, "rank mismatch")
		return linearIndex(of: offset) + shape.elementSpanCount <= elementSpanCount
	}

	//----------------------------------------------------------------------------
	// ofItems
	public func ofItems(count: Int) -> Shape {
		return Shape(extent: [count] + extent.suffix(from: 1),
		             layout: layout,
		             strides: strides, colMajor: colMajor)
	}
	
	//----------------------------------------------------------------------------
	// BinarySerializable
	public init(from buffer: BufferUInt8, next: inout BufferUInt8) {
		layout = DataLayout(from: buffer, next: &next)
		channelFormat = ChannelFormat(from: next, next: &next)
		extent = [Int](from: next, next: &next)
		strides = [Int](from: next, next: &next)
		colMajor = Bool(from: next, next: &next)
		elementCount = extent.reduce(1, *)
		elementSpanCount = Shape.getSpanCount(extent: extent, strides: strides)
		rank = extent.count
		channelsAxis = layout.channelsAxis
		rowsAxis = layout.rowsAxis
		colsAxis = layout.colsAxis
	}
	
	public func serialize(to buffer: inout [UInt8]) {
		layout.serialize(to: &buffer)
		channelFormat.serialize(to: &buffer)
		extent.serialize(to: &buffer)
		strides.serialize(to: &buffer)
		colMajor.serialize(to: &buffer)
	}
	
	//----------------------------------------------------------------------------
	// operator ==
	public static func ==(lhs: Shape, rhs: Shape) -> Bool {
		return lhs.elementCount == rhs.elementCount &&
			lhs.extent == rhs.extent && lhs.strides == rhs.strides
	}
	
	//----------------------------------------------------------------------------
	// computeStrides
	private static func computeStrides(for dims: [Int]) -> [Int] {
		var strides = [Int](repeating: 1, count: dims.count)
		for dim in (0..<dims.count - 1).reversed() {
			strides[dim] = dims[dim + 1] * strides[dim + 1]
		}
		return strides
	}
	
	//----------------------------------------------------------------------------
	// getSpanCount
	// a sub view may cover a wider range of parent element indexes
	// than the number of elements defined by the extent of this view
	// The span of the extent is the linear index of the last index + 1
	private static func getSpanCount(extent: [Int], strides: [Int]) -> Int {
		assert(extent.count == strides.count)
		var spanCount = 1
		for i in 0..<extent.count {
			spanCount += (extent[i] - 1) * strides[i]
		}
		return spanCount
	}
}

//==============================================================================
// ChannelFormat
//	A value of 'None' indicates no predefined meaning for a sample channel
public enum ChannelFormat : String, EnumerableType {
	case any, gray, grayAlpha, rgb, rgba

	public var channels: Int {
		switch self {
		case .any      : fatalError("invalid operation")
		case .gray     : return 1
		case .grayAlpha: return 2
		case .rgb      : return 3
		case .rgba     : return 4
		}
	}
}

//==============================================================================
// forEachIndex
extension DataView {
	public func forEachIndex(fn: (Int) -> Void) {
		switch rank {
		case 1:
			if shape.isContiguous {
				for i in 0..<shape.items { fn(i) }
			} else {
				var linearItemA = 0
				for _ in 0..<shape.items {
					fn(linearItemA)
					linearItemA += shape.itemStride
				}
			}
			
		case 2:
			var linearRowA = 0
			for _ in 0..<shape.rows {
				var linearColA = linearRowA
				for _ in 0..<shape.cols {
					fn(linearColA)
					linearColA += shape.colStride
				}
				linearRowA += shape.rowStride
			}
			
		//--------------------------------------------------------------------------
		case 4:
			// walk 4D tensor
			var linearItemA = 0
			for _ in 0..<shape.items {
				var linearRowA = linearItemA

				for _ in 0..<shape.rows {
					var linearColA = linearRowA

					for _ in 0..<shape.cols {
						var linearChanA = linearColA

						for _ in 0..<shape.channels {
							fn(linearChanA)
							linearChanA += shape.channelStride
						}
						linearColA += shape.colStride
					}
					linearRowA += shape.rowStride
				}
				linearItemA += shape.itemStride
			}

		//--------------------------------------------------------------------------
		default: fatalError("not implemented yet")
		}
	}

	//============================================================================
	// forEachIndex(with other
	public func forEachIndex(with other: DataView, fn: (Int, Int) -> Void) {
		// validate
		assert(rank == other.rank)
		assert(shape.items == other.items)

		switch rank {
		case 1:
			if shape.isContiguous && other.shape.isContiguous {
				for i in 0..<shape.items { fn(i, i) }
			} else {
				var linearItemA = 0, linearItemB = 0
				for _ in 0..<shape.items {
					fn(linearItemA, linearItemB)
					linearItemA += shape.itemStride
					linearItemB += other.shape.itemStride
				}
			}
			
		case 2:
			assert(shape.rows == other.rows && shape.cols == other.cols)
			var linearRowA = 0
			var linearRowB = 0
			for _ in 0..<shape.rows {
				var linearColA = linearRowA
				var linearColB = linearRowB
				for _ in 0..<shape.cols {
					fn(linearColA, linearColB)
					linearColA += shape.colStride
					linearColB += other.shape.colStride
				}
				linearRowA += shape.rowStride
				linearRowB += other.shape.rowStride
			}
			
		//--------------------------------------------------------------------------
		case 4:
			assert(shape.channels == other.channels &&
				shape.rows == other.rows && shape.cols == other.cols)
			
			// walk 4D tensor samples
			var linearItemA = 0, linearItemB = 0
			for _ in 0..<shape.items {
				var linearRowA = linearItemA
				var linearRowB = linearItemB
				
				for _ in 0..<shape.rows {
					var linearColA = linearRowA
					var linearColB = linearRowB
					
					for _ in 0..<shape.cols {
						var linearChanA = linearColA
						var linearChanB = linearColB
						
						for _ in 0..<shape.channels {
							fn(linearChanA, linearChanB)
							linearChanA += shape.channelStride
							linearChanB += other.shape.channelStride
						}
						
						linearColA += shape.colStride
						linearColB += other.shape.colStride
					}
					linearRowA += shape.rowStride
					linearRowB += other.shape.rowStride
				}
				linearItemA += shape.itemStride
				linearItemB += other.shape.itemStride
			}

		//--------------------------------------------------------------------------
		default: fatalError("not implemented yet")
		}
	}
}






