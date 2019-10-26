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
import ImageCodecs

final public class ImageCodec : ModelObjectBase, Codec, InitHelper {
	//----------------------------------------------------------------------------
	// properties
	public var format = ImageFormat()	                 { didSet{onSet("format")} }
	public var codecType: CodecType { return .image }
	private static let impliedFormat: [ChannelFormat] = [.any, .gray, .grayAlpha, .rgb, .rgba]

	//----------------------------------------------------------------------------
	// addAccessors
	public override func addAccessors() {
		super.addAccessors()
		addAccessor(name: "format",
		            get: { [unowned self] in self.format },
		            set: { [unowned self] in self.format = $0 })
	}

	//----------------------------------------------------------------------------
	// encodedShape(of:
	public func encodedShape(of shape: Shape) -> Shape {
		guard format.channelFormat != .any else { return shape }
		// determine output channelFormat
		let channelFormat = format.channelFormat == .any ?
			ImageCodec.impliedFormat[shape.channels] : format.channelFormat

		return Shape(extent: [1, shape.rows, shape.cols, format.channelFormat.channels],
		             layout: .nhwc, channelFormat: channelFormat)
	}

	//----------------------------------------------------------------------------
	// decode
	public func decode(buffer: BufferUInt8, to outData: inout DataView) throws {
		guard let encoding = detectImageEncoding(buffer: buffer) else {
			writeLog("ImageCodec.decode - unrecognized data encoding")
			throw ModelError.conversationFailed("")
		}

		switch encoding {
		//----------------------------------
		case .png:
			// find out the type and size of the output
			let info = pngDecodeInfo(buffer.baseAddress, buffer.count,
			                         format.channelFormat.ctype)
			defer { freePngCDecoderInfo(info) }
			
			let dataType = DataType(type: info.dataType)
			let dims = [Int](UnsafeBufferPointer(start: info.dims, count: info.numDims))
			let channelFormat = ImageCodec.impliedFormat[dims[2]]
			let shape = Shape(extent: [1] + dims, layout: .nhwc,
			                  channelFormat: channelFormat)
			assert(shape.elementCount == outData.shape.elementCount)
			
			// decode to native
			var native = DataView(shape: shape, dataType: dataType)
			let buffer = try native.rwReal8U()
			pngDecodeData(info, buffer.baseAddress!, buffer.count)
			try cpuCopy(from: native, to: &outData, normalizeInts: true)
			
		//----------------------------------
		case .jpeg:
			// find out the type and size of the output
			let info = jpegDecodeInfo(buffer.baseAddress, buffer.count,
			                          format.channelFormat.ctype)
			defer { freeJpegCDecoderInfo(info) }
			
			let dataType = DataType(type: info.dataType)
			let dims = [Int](UnsafeBufferPointer(start: info.dims, count: info.numDims))
			let channelFormat = ImageCodec.impliedFormat[dims[2]]
			let shape = Shape(extent: [1] + dims, layout: .nhwc,
			                  channelFormat: channelFormat)
			assert(shape.elementCount == outData.shape.elementCount)
			
			// decode to native
			var native = DataView(shape: shape, dataType: dataType)
			let buffer = try native.rwReal8U()
			jpegDecodeData(info, buffer.baseAddress!, buffer.count)
			try cpuCopy(from: native, to: &outData, normalizeInts: true)
			
		default: fatalError("not implemented")
		}
	}
	
	//----------------------------------------------------------------------------
	// decodeInfo
	public func decodeInfo(buffer: BufferUInt8) throws -> (DataType, Shape) {
		guard let encoding = detectImageEncoding(buffer: buffer) else {
			writeLog("ImageCodec.decodeInfo - unrecognized data encoding")
			throw ModelError.conversationFailed("")
		}
		
		switch encoding {
		case .png:
			// find out the type and size of the output
			let info = pngDecodeInfo(buffer.baseAddress, buffer.count,
			                         format.channelFormat.ctype)
			defer { freePngCDecoderInfo(info) }
			let dataType = DataType(type: info.dataType)
			let dims = [Int](UnsafeBufferPointer(start: info.dims, count: info.numDims))
			let channelFormat = ImageCodec.impliedFormat[dims[2]]
			let shape = Shape(extent: [1] + dims, layout: .nhwc,
			                  channelFormat: channelFormat)
			return (dataType, shape)
			
		case .jpeg:
			// find out the type and size of the output
			let info = jpegDecodeInfo(buffer.baseAddress, buffer.count,
			                          format.channelFormat.ctype)
			defer { freeJpegCDecoderInfo(info) }
			let dataType = DataType(type: info.dataType)
			let dims = [Int](UnsafeBufferPointer(start: info.dims, count: info.numDims))
			let channelFormat = ImageCodec.impliedFormat[dims[2]]
			let shape = Shape(extent: [1] + dims, layout: .nhwc,
			                  channelFormat: channelFormat)
			return (dataType, shape)
			
		default: fatalError("not implemented")
		}
	}
	
	//----------------------------------------------------------------------------
	// encode
	public func encode(data: DataView,
	                   using stream: DeviceStream? = nil,
	                   completion: EncodedHandler) throws {
		assert(data.shape.items == 1 && data.shape.channels <= 4)
		guard format.encoding != .any else {
			writeLog("ImageCodec::encode - an encoding type must be specified")
			throw ModelError.conversationFailed("")
		}
		
		// determine output channelFormat
		let channelFormat = format.channelFormat == .any ?
			ImageCodec.impliedFormat[data.channels] : format.channelFormat

		// ensure the correct image type and layout
		let dims = [data.shape.rows, data.shape.cols, channelFormat.channels]
		let shape = Shape(extent: [1] + dims, layout: .nhwc,
		                  channelFormat: channelFormat)

		var source: DataView
		if data.shape == shape && data.dataType == format.dataType {
			source = data
		} else {
			source = DataView(shape: shape, dataType: format.dataType)
			try cpuConvertImage(normalizedInData: data, to: &source)
		}

		switch format.encoding {
		//----------------------------------
		case .png:
			let buffer  = try source.roReal8U()
			let encoded = pngEncode(buffer.baseAddress!, buffer.count,
			                        dims, dims.count, source.dataType.ctype,
			                        format.compression)
			defer { freeCByteArray(encoded) }
			let encodedBuffer = UnsafeBufferPointer(start: encoded.data,
			                                        count: encoded.count)
			try completion(source.dataType, shape, encodedBuffer)
			
		//----------------------------------
		case .jpeg:
			let buffer  = try source.roReal8U()
			let encoded = jpegEncode(buffer.baseAddress!, buffer.count,
			                         dims, dims.count, format.jpegQuality)
			defer { freeCByteArray(encoded) }
			let encodedBuffer = UnsafeBufferPointer(start: encoded.data,
			                                        count: encoded.count)
			try completion(source.dataType, shape, encodedBuffer)
			
		default: fatalError("not implemented")
		}
	}
	
	//----------------------------------------------------------------------------
	// recode
	//	This assures the output data is in the correct format, decoding and
	// encoding if necessary
	//
	public func recode(buffer: BufferUInt8, using stream: DeviceStream? = nil,
	                   completion: EncodedHandler) throws {
		guard let encoding = detectImageEncoding(buffer: buffer) else {
			writeLog("ImageCodec.recode - unrecognized data encoding")
			throw ModelError.conversationFailed("")
		}
		let outputEncoding = format.encoding == .any ? encoding : format.encoding

		switch encoding {
		//----------------------------------
		case .png:
			// find out the type and size of the output
			let info = pngDecodeInfo(buffer.baseAddress, buffer.count,
			                         format.channelFormat.ctype)
			defer { freePngCDecoderInfo(info) }
			let dataType = DataType(type: info.dataType)
			let dims = [Int](UnsafeBufferPointer(start: info.dims, count: info.numDims))

			// set output shape and channelFormat
			let shape = Shape(extent: [1] + dims, layout: .nhwc,
				                channelFormat: ImageCodec.impliedFormat[dims[2]])
			
			// if it's already in the correct encoding then we're done
			if info.willChangeType == 0 && format.encoding == .png ||
				 format.encoding == .any {
				
				try completion(dataType, shape, buffer)
				return
			} else {
				// recode
				var data   = DataView(shape: shape, dataType: dataType)
				let buffer = try data.rwReal8U()
				pngDecodeData(info, buffer.baseAddress!, buffer.count)
				try encode(data: data, completion: completion)
			}
			
		//----------------------------------
		case .jpeg:
			// find out the type and size of the output
			let info = jpegDecodeInfo(buffer.baseAddress, buffer.count,
			                          format.channelFormat.ctype)
			defer { freeJpegCDecoderInfo(info) }
			let dataType = DataType(type: info.dataType)
			let dims = [Int](UnsafeBufferPointer(start: info.dims, count: info.numDims))

			// set output shape and channelFormat
			let shape = Shape(extent: [1] + dims, layout: .nhwc,
				                channelFormat: ImageCodec.impliedFormat[dims[2]])
			
			// if it's already in the correct encoding then we're done
			if info.willChangeType == 0 && format.encoding == .jpeg ||
				 format.encoding == .any {

				try completion(dataType, shape, buffer)
				return
			} else {
				// recode
				var data   = DataView(shape: shape, dataType: dataType)
				let buffer = try data.rwReal8U()
				jpegDecodeData(info, buffer.baseAddress!, buffer.count)
				try encode(data: data, completion: completion)
			}
			
		default: fatalError("not implemented")
		}
	}
} // ImageCodec

//------------------------------------------------------------------------------
// detectImageEncoding
public func detectImageEncoding(buffer: BufferUInt8) -> ImageEncoding? {
	// png
	if (buffer.baseAddress!.withMemoryRebound(to: UInt.self, capacity: 1) {
		return $0[0] == UInt(bigEndian: 0x89504E470D0A1A0A)
	}) { return .png }

	// jpeg   TODO this isn't the whole signature, figure out "nn"
	if (buffer.baseAddress!.withMemoryRebound(to: UInt32.self, capacity: 1) {
		return ($0[0] == UInt32(bigEndian: UInt32(0xFFD8FFDB)) ||
			$0[0] == UInt32(bigEndian: UInt32(0xFFD8FFE0)) ||
			$0[0] == UInt32(bigEndian: UInt32(0xFFD8FFE1)))
	}) { return .jpeg }

	return nil
}



