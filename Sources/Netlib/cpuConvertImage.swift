//******************************************************************************
//  Created by Edward Connell on 10/7/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
// https://en.wikipedia.org/wiki/SRGB
// http://stackoverflow.com/questions/15686277/convert-rgb-to-grayscale-in-c
//
import Foundation

// TODO: I wish .Type could be passed and used as a variable. The code would
// be so much smaller

public func cpuConvertImage(normalizedInData inData: DataView,
                            to outData: inout DataView) throws {
	// validate
	assert(inData.items == 1 && outData.items == 1 &&
		inData.rows == outData.rows && inData.cols == outData.cols)

	// simple copy if they have the same channel format. cpuCopy takes care of
	// possible stride, format, and dataType conversions
	if inData.shape.channelFormat == outData.shape.channelFormat {
		try cpuCopy(from: inData, to: &outData, normalizeInts: true)
		return
	}

	//-------------------------------------
	// convert
	func convert<T: AnyNumber, U: AnyNumber>(_ t: T.Type, _ u: U.Type) throws {
		
		switch inData.shape.channelFormat {
		case .gray:
			switch outData.shape.channelFormat {
			case .grayAlpha:
				try outData.forEachMutableSample(with: inData) {
					(dst: inout MutableGrayAlphaSample<T>, src: GraySample<U>) in
					dst.gray = T(norm: src.gray)
					dst.alpha = T(norm: 1)
				}
				
			case .rgb:
				try outData.forEachMutableSample(with: inData) {
					(dst: inout MutableRGBSample<T>, src: GraySample<U>) in
					dst.r = T(norm: src.gray)
					dst.g = dst.r
					dst.b = dst.r
				}
				
			case .rgba:
				try outData.forEachMutableSample(with: inData) {
					(dst: inout MutableRGBASample<T>, src: GraySample<U>) in
					dst.r = T(norm: src.gray)
					dst.g = dst.r
					dst.b = dst.r
					dst.a = T(norm: 1)
				}
				
			default: fatalError("\(inData.shape.channelFormat) to " +
				"\(outData.shape.channelFormat) not implemented")
			}
			
		case .grayAlpha:
			switch outData.shape.channelFormat {
			case .gray:
				try outData.forEachMutableSample(with: inData) {
					(dst: inout MutableGraySample<T>, src: GrayAlphaSample<U>) in
					dst.gray = T(norm: src.gray)
				}
				
			case .rgb:
				try outData.forEachMutableSample(with: inData) {
					(dst: inout MutableRGBSample<T>, src: GrayAlphaSample<U>) in
					dst.r = T(norm: src.gray)
					dst.g = dst.r
					dst.b = dst.r
				}
				
			case .rgba:
				try outData.forEachMutableSample(with: inData) {
					(dst: inout MutableRGBASample<T>, src: GrayAlphaSample<U>) in
					dst.r = T(norm: src.gray)
					dst.g = dst.r
					dst.b = dst.r
					dst.a = T(norm: src.alpha)
				}
				
			default: fatalError("\(inData.shape.channelFormat) to " +
				"\(outData.shape.channelFormat) not implemented")
			}
			
		case .rgb:
			switch outData.shape.channelFormat {
			case .gray:
				try outData.forEachMutableSample(with: inData) {
					(dst: inout MutableGraySample<T>, src: RGBSample<U>) in
					dst.gray = T(norm: sRGBToGray(r: src.r.normDouble,
					                              g: src.g.normDouble,
					                              b: src.b.normDouble))
				}
				
			case .grayAlpha:
				try outData.forEachMutableSample(with: inData) {
					(dst: inout MutableGrayAlphaSample<T>, src: RGBSample<U>) in
					dst.gray = T(norm: sRGBToGray(r: src.r.normDouble,
					                              g: src.g.normDouble,
					                              b: src.b.normDouble))
					dst.alpha = T(norm: 1)
				}

			case .rgba:
				try outData.forEachMutableSample(with: inData) {
					(dst: inout MutableRGBASample<T>, src: RGBSample<U>) in
					dst.r = T(norm: src.r)
					dst.g = T(norm: src.g)
					dst.b = T(norm: src.b)
					dst.a = T(norm: 1)
				}
				
			default: fatalError("\(inData.shape.channelFormat) to " +
				"\(outData.shape.channelFormat) not implemented")
			}
			
		case .rgba:
			switch outData.shape.channelFormat {
			case .gray:
				try outData.forEachMutableSample(with: inData) {
					(dst: inout MutableGraySample<T>, src: RGBASample<U>) in
					dst.gray = T(norm: sRGBToGray(r: src.r.normDouble,
					                              g: src.g.normDouble,
					                              b: src.b.normDouble))
					// TODO: should alpha be stripped or applied?
//					dst.gray = T(any: gray.asDouble * src.a.normDouble)
				}
				
			case .grayAlpha:
				try outData.forEachMutableSample(with: inData) {
					(dst: inout MutableGrayAlphaSample<T>, src: RGBASample<U>) in
					dst.gray = T(norm: sRGBToGray(r: src.r.normDouble,
					                              g: src.g.normDouble,
					                              b: src.b.normDouble))
					dst.alpha = T(norm: src.a)
				}
				
			case .rgb:
				try outData.forEachMutableSample(with: inData) {
					(dst: inout MutableRGBASample<T>, src: RGBASample<U>) in
					dst.r = T(norm: src.r)
					dst.g = T(norm: src.g)
					dst.b = T(norm: src.b)
				}

			default: fatalError("\(inData.shape.channelFormat) to " +
				"\(outData.shape.channelFormat) not implemented")
			}
		default: fatalError()
		}
	}
	
	//-------------------------------------
	// specialize
	switch outData.dataType {
	case .real8U:
		switch inData.dataType {
		case .real8U:  try convert(UInt8.self, UInt8.self)
		case .real16U: try convert(UInt8.self, UInt16.self)
		case .real16I: try convert(UInt8.self, Int16.self)
		case .real32I: try convert(UInt8.self, Int32.self)
		case .real16F: try convert(UInt8.self, Float16.self)
		case .real32F: try convert(UInt8.self, Float.self)
		case .real64F: try convert(UInt8.self, Double.self)
		}
		
	case .real16U:
		switch inData.dataType {
		case .real8U:  try convert(UInt16.self, UInt8.self)
		case .real16U: try convert(UInt16.self, UInt16.self)
		case .real16I: try convert(UInt16.self, Int16.self)
		case .real32I: try convert(UInt16.self, Int32.self)
		case .real16F: try convert(UInt16.self, Float16.self)
		case .real32F: try convert(UInt16.self, Float.self)
		case .real64F: try convert(UInt16.self, Double.self)
		}
		
	case .real16I:
		switch inData.dataType {
		case .real8U:  try convert(Int16.self, UInt8.self)
		case .real16U: try convert(Int16.self, UInt16.self)
		case .real16I: try convert(Int16.self, Int16.self)
		case .real32I: try convert(Int16.self, Int32.self)
		case .real16F: try convert(Int16.self, Float16.self)
		case .real32F: try convert(Int16.self, Float.self)
		case .real64F: try convert(Int16.self, Double.self)
		}

	case .real32I:
		switch inData.dataType {
		case .real8U:  try convert(Int32.self, UInt8.self)
		case .real16U: try convert(Int32.self, UInt16.self)
		case .real16I: try convert(Int32.self, Int16.self)
		case .real32I: try convert(Int32.self, Int32.self)
		case .real16F: try convert(Int32.self, Float16.self)
		case .real32F: try convert(Int32.self, Float.self)
		case .real64F: try convert(Int32.self, Double.self)
		}

	case .real16F:
		switch inData.dataType {
		case .real8U:  try convert(Float16.self, UInt8.self)
		case .real16U: try convert(Float16.self, UInt16.self)
		case .real16I: try convert(Float16.self, Int16.self)
		case .real32I: try convert(Float16.self, Int32.self)
		case .real16F: try convert(Float16.self, Float16.self)
		case .real32F: try convert(Float16.self, Float.self)
		case .real64F: try convert(Float16.self, Double.self)
		}
		
	case .real32F:
		switch inData.dataType {
		case .real8U:  try convert(Float.self, UInt8.self)
		case .real16U: try convert(Float.self, UInt16.self)
		case .real16I: try convert(Float.self, Int16.self)
		case .real32I: try convert(Float.self, Int32.self)
		case .real16F: try convert(Float.self, Float16.self)
		case .real32F: try convert(Float.self, Float.self)
		case .real64F: try convert(Float.self, Double.self)
		}
		
	case .real64F:
		switch inData.dataType {
		case .real8U:  try convert(Double.self, UInt8.self)
		case .real16U: try convert(Double.self, UInt16.self)
		case .real16I: try convert(Double.self, Int16.self)
		case .real32I: try convert(Double.self, Int32.self)
		case .real16F: try convert(Double.self, Float16.self)
		case .real32F: try convert(Double.self, Float.self)
		case .real64F: try convert(Double.self, Double.self)
		}
	}
}

//==============================================================================
// Utility functions

// sRGBToLinearColorSpace
public func sRGBToLinearColorSpace(_ value: Double) -> Double {
	return value < 0.04045 ? (value / 12.92) : pow( (value + 0.055) / 1.055, 2.4)
}

// LinearColorSpaceTosRGB
public func LinearColorSpaceTosRGB(_ value: Double) -> Double {
	return value <= 0.0031308 ?	(12.92 * value) : (1.055 * pow(value, 1/2.4) - 0.055)
}

// sRGBToGray
public func sRGBToGray(r: Double, g: Double, b: Double) -> Double {
	let grayLinear =
		0.2126 * sRGBToLinearColorSpace(r) +
			0.7152 * sRGBToLinearColorSpace(g) +
			0.0722 * sRGBToLinearColorSpace(b)
	return LinearColorSpaceTosRGB(grayLinear)
}

