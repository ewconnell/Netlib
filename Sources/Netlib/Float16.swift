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

public struct Float16 : Equatable, Comparable {
	// initializers
	public init() {	x = UInt16(0) }
	public init(_ v: Float16) { x = v.x }
	public init(bitPattern: UInt16) { x = bitPattern }
	public init?(_ string: String) {
		if let v = Float(string) {
			x = floatToFloat16_rn(v).x
		} else {
			return nil
		}
	}

	public init(_ v: UInt8)  { x = floatToFloat16_rn(Float(v)).x	}
	public init(_ v: UInt16) { x = floatToFloat16_rn(Float(v)).x	}
	public init(_ v: Int16)  { x = floatToFloat16_rn(Float(v)).x	}
	public init(_ v: Int32)  { x = floatToFloat16_rn(Float(v)).x	}
	public init(_ v: Int)    { x = floatToFloat16_rn(Float(v)).x	}
	public init(_ v: Float)  { x = floatToFloat16_rn(v).x	}
	public init(_ d: Double) { x = floatToFloat16_rn(Float(d)).x }
	public init(d: Double)   { x = floatToFloat16_rn(Float(d)).x }
	
	// properties
	var x: UInt16
	
	// 10:5:1
	private static let mantissaMask: UInt16 = 0b0000001111111111
	private static let exponentMask: UInt16 = 0b0111110000000000
	private static let signMask:     UInt16 = 0b1000000000000000
	
	// functions
	public var mantissa: Int {
		get {	return (Int)(x & Float16.mantissaMask) }
	}
	public var exponent: Int {
		get {	return (Int)(x & Float16.exponentMask) }
	}
	public var sign: Int {
		get {	return (Int)(x & Float16.signMask) }
	}
	
	public static func <(lhs: Float16, rhs: Float16) -> Bool {
		return Float(lhs) < Float(rhs)
	}
	
	public static func ==(lhs: Float16, rhs: Float16) -> Bool {
		return lhs.x == rhs.x
	}
	
	// operators
	public static func +(lhs: Float16, rhs: Float16) -> Float16 {
		return Float16(Float(lhs) + Float(rhs))
	}
	
	public static func -(lhs: Float16, rhs: Float16) -> Float16 {
		return Float16(Float(lhs) - Float(rhs))
	}

	public static func *(lhs: Float16, rhs: Float16) -> Float16 {
		return Float16(Float(lhs) * Float(rhs))
	}

	public static func /(lhs: Float16, rhs: Float16) -> Float16 {
		return Float16(Float(lhs) / Float(rhs))
	}
}

//==============================================================================
// helpers
public func habs(_ h: Float16) -> Float16 {
 return Float16(bitPattern: h.x & UInt16(0x7fff))
}

public func hneg(_ h: Float16) -> Float16 {
 return Float16(bitPattern: h.x ^ UInt16(0x8000))
}

public func ishnan(_ h: Float16) -> Bool {
	// When input is NaN, exponent is all 1s and mantissa is non-zero.
	return (h.x & UInt16(0x7c00)) == UInt16(0x7c00) && (h.x & UInt16(0x03ff)) != 0
}

public func ishinf(_ h: Float16) -> Bool {
	// When input is +/- inf, exponent is all 1s and mantissa is zero.
	return (h.x & UInt16(0x7c00)) == UInt16(0x7c00) && (h.x & UInt16(0x03ff)) == 0

}

public func ishequ(x: Float16, y: Float16) -> Bool {
	return !ishnan(x) && !ishnan(y) && x.x == y.x
}

public func hzero() -> Float16 { return Float16() }

public func hone() -> Float16 { return Float16(bitPattern: UInt16(0x3c00)) }

//==============================================================================
// extensions
extension Float {
	public init(_ fp16: Float16) { self = float16ToFloat(fp16) }
}

extension UInt8 {
	public init(_ fp16: Float16) { self = UInt8(Float(fp16)) }
}

extension UInt16 {
	public init(_ fp16: Float16) { self = UInt16(Float(fp16)) }
}

extension Int16 {
	public init(_ fp16: Float16) { self = Int16(Float(fp16)) }
}

extension Int32 {
	public init(_ fp16: Float16) { self = Int32(Float(fp16)) }
}

extension Int {
	public init(_ fp16: Float16) { self = Int(Float(fp16)) }
}

extension Double {
	public init(_ fp16: Float16) { self = Double(Float(fp16)) }
}

//==============================================================================
// floatToFloat16_rn
//	cpu functions for converting between FP32 and FP16 formats
// inspired from Paulius Micikevicius (pauliusm@nvidia.com)

public func floatToFloat16_rn(_ f: Float) -> Float16 {
	var result = Float16()
	
	let x = f.bitPattern
	let u: UInt32 = x & 0x7fffffff
	var remainder, shift, lsb, lsb_s1, lsb_m1:UInt32
	var sign, exponent, mantissa: UInt32
	
	// Get rid of +NaN/-NaN case first.
	if (u > 0x7f800000) {
		result.x = UInt16(0x7fff)
		return result;
	}
	
	sign = ((x >> 16) & UInt32(0x8000))
	
	// Get rid of +Inf/-Inf, +0/-0.
	if (u > 0x477fefff) {
		result.x = UInt16(sign | UInt32(0x7c00))
		return result
	}
	if (u < 0x33000001) {
		result.x = UInt16(sign | 0x0000)
		return result
	}
	
	exponent = ((u >> 23) & 0xff)
	mantissa = (u & 0x7fffff)
	
	if (exponent > 0x70) {
		shift = 13
		exponent -= 0x70
	} else {
		shift = 0x7e - exponent
		exponent = 0
		mantissa |= 0x800000
	}
	lsb    = (1 << shift)
	lsb_s1 = (lsb >> 1)
	lsb_m1 = (lsb - 1)
	
	// Round to nearest even.
	remainder = (mantissa & lsb_m1)
	mantissa >>= shift
	if (remainder > lsb_s1 || (remainder == lsb_s1 && (mantissa & 0x1) != 0)) {
		mantissa += 1
		if ((mantissa & 0x3ff) == 0) {
			exponent += 1
			mantissa = 0
		}
	}
	
	result.x = UInt16(sign | (exponent << 10) | mantissa)
	return result
}


//==============================================================================
// float16ToFloat
public func float16ToFloat(_ h: Float16) -> Float
{
	var sign     = UInt32((h.x >> 15) & 1)
	var exponent = UInt32((h.x >> 10) & 0x1f)
	var mantissa = UInt32(h.x & 0x3ff) << 13
	
	if exponent == 0x1f {  /* NaN or Inf */
		if mantissa != 0 {
			sign = 0
			mantissa = UInt32(0x7fffff)
		} else {
			mantissa = 0
		}
		exponent = 0xff
	} else if exponent == 0 {  /* Denorm or Zero */
		if mantissa != 0 {
			var msb: UInt32
			exponent = 0x71
			repeat {
				msb = (mantissa & 0x400000)
				mantissa <<= 1  /* normalize */
				exponent -= 1
			} while msb == 0
			mantissa &= 0x7fffff  /* 1.mantissa is implicit */
		}
	} else {
		exponent += 0x70
	}
	
	return Float(bitPattern: UInt32((sign << 31) | (exponent << 23) | mantissa))
}














