//******************************************************************************
//  Created by Edward Connell on 9/8/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
import Foundation

//==============================================================================
// cpuFill
public func cpuFill(data: inout DataView, with constant: Double) throws {
	if constant == 0 && data.shape.isContiguous {
		let buffer = try data.rwReal8U()
		for i in 0..<buffer.count {	buffer[i] = 0 }
	} else {

		func setValues<T: AnyNumber>(
			_ result: inout DataView, _ type: T.Type) throws {

			let value = T(any: constant)
			let buffer = try result.rw(type: T.self)
			result.forEachIndex { buffer[$0] = value }
		}

		switch data.dataType {
		case .real8U:  try setValues(&data, UInt8.self)
		case .real16U: try setValues(&data, UInt16.self)
		case .real16I: try setValues(&data, Int16.self)
		case .real32I: try setValues(&data, Int32.self)
		case .real16F: try setValues(&data, Float16.self)
		case .real32F: try setValues(&data, Float.self)
		case .real64F: try setValues(&data, Double.self)
		}
	}
}

//==============================================================================
// cpuFillUniform
public func cpuFillUniform(data: inout DataView, range: ClosedRange<Double>) throws {

	func setValues<T: AnyNumber>(_ result: inout DataView, _ type: T.Type) throws {
		let rng = Xoroshiro128Plus()
		let buffer = try result.rw(type: T.self)
		result.forEachIndex { buffer[$0] = T(norm: rng.next().double(inRange: range)) }
	}

	switch data.dataType {
	case .real8U:  try setValues(&data, UInt8.self)
	case .real16U: try setValues(&data, UInt16.self)
	case .real16I: try setValues(&data, Int16.self)
	case .real32I: try setValues(&data, Int32.self)
	case .real16F: try setValues(&data, Float16.self)
	case .real32F: try setValues(&data, Float.self)
	case .real64F: try setValues(&data, Double.self)
	}
}

//==============================================================================
// computeVarianceNorm
// The input data must have the shape(num, chans, rows, cols) where
//  fan_in = chans * rows * cols
//  fan_out = items * rows * cols
public func computeVarianceNorm(shape: Shape, varianceNorm: FillVarianceNorm) -> Double {
	let fanIn, fanOut: Double
	let eltCount = Double(shape.elementCount)

	switch shape.rank {
	case 1:
		fanIn  = Double(shape.items)
		fanOut = fanIn
	case 2:
		fanIn  = Double(shape.rows)
		fanOut = Double(shape.cols)

	default:
		fanIn  = eltCount / Double(shape.items)
		fanOut = Double(shape.items * shape.rows * shape.cols)
	}

	var n: Double
	switch varianceNorm {
	case .fanIn:   n = fanIn
	case .fanOut:  n = fanOut
	case .average: n = (fanIn + fanOut) / 2
	}

	return n
}

//==============================================================================
// cpuFillXavier
//
//	A Filler based on the paper[Bengio and Glorot 2010]: Understanding
// the difficulty of training deep feed forward neural networks.
//
// It fills the incoming matrix by randomly sampling uniform data from[-scale,
// scale] where scale = sqrt(3 / n) where n is the fan_in, fan_out, or their
// average, depending on the variance_norm option.
//
// http://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization
//
public func cpuFillXavier(data: inout DataView, varianceNorm: FillVarianceNorm,
                          seed: UInt? = nil) throws {
	// maybe support non contiguous later
	assert(data.shape.isContiguous)
	let n = computeVarianceNorm(shape: data.shape, varianceNorm: varianceNorm)

	// generate rands in the range of [-range, +range]
	func setValues<T: AnyNumber>(_ result: inout DataView, _ type: T.Type) throws {
		let randSeed = seed != nil ? UInt64(seed!) : UInt64(Date().timeIntervalSince1970)
		let rng = Xoroshiro128Plus(seed: randSeed)
		let range = sqrt(3.0 / n) * 2
		let buffer = try result.rw(type: T.self)
		result.forEachIndex {
			buffer[$0] = T(norm: (rng.next().doubleInUnitRange() - 0.5) * range)
		}
	}
	
	switch data.dataType {
	case .real8U:  try setValues(&data, UInt8.self)
	case .real16U: try setValues(&data, UInt16.self)
	case .real16I: try setValues(&data, Int16.self)
	case .real32I: try setValues(&data, Int32.self)
	case .real16F: try setValues(&data, Float16.self)
	case .real32F: try setValues(&data, Float.self)
	case .real64F: try setValues(&data, Double.self)
	}
}

//==============================================================================
// cpuFillWithIndex
//	This is mainly useful for debugging purposes
public func cpuFillWithIndex(data: inout DataView, startingAt: Int) throws {
	var index = startingAt
	func setValues<T: AnyNumber>(_ result: inout DataView, _ type: T.Type) throws {
		let buffer = try result.rw(type: T.self)
		result.forEachIndex {
			buffer[$0] = T(any: index)
			index += 1
		}
	}
	
	switch data.dataType {
	case .real8U:  try setValues(&data, UInt8.self)
	case .real16U: try setValues(&data, UInt16.self)
	case .real16I: try setValues(&data, Int16.self)
	case .real32I: try setValues(&data, Int32.self)
	case .real16F: try setValues(&data, Float16.self)
	case .real32F: try setValues(&data, Float.self)
	case .real64F: try setValues(&data, Double.self)
	}
}

//==============================================================================
// cpuFillGaussian
public func cpuFillGaussian(data: inout DataView, mean: Double, std: Double) throws {

	func setValues<T: AnyNumber>(_ result: inout DataView, _ type: T.Type) throws {
		let buffer = try result.rw(type: T.self)
		let rng = Xoroshiro128Plus()

		result.forEachIndex {
			buffer[$0] = T(norm: mean + rng.nextGaussianInUnitRange() * std)
		}
	}
	
	switch data.dataType {
	case .real8U:  try setValues(&data, UInt8.self)
	case .real16U: try setValues(&data, UInt16.self)
	case .real16I: try setValues(&data, Int16.self)
	case .real32I: try setValues(&data, Int32.self)
	case .real16F: try setValues(&data, Float16.self)
	case .real32F: try setValues(&data, Float.self)
	case .real64F: try setValues(&data, Double.self)
	}
}

//==============================================================================
// cpuFillMSRA
public func cpuFillMSRA(data: inout DataView, varianceNorm: FillVarianceNorm) throws {
	let n = computeVarianceNorm(shape: data.shape, varianceNorm: varianceNorm)
	try cpuFillGaussian(data: &data, mean: 0, std: sqrt(2.0 / n))
}




