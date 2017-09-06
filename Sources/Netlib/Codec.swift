//******************************************************************************
//  Created by Edward Connell on 6/2/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
import ImageCodecs

//==============================================================================
// Codec
public protocol Codec : ModelObject {
	// properties
	var codecType : CodecType { get }

	// functions
	func encodedShape(of: Shape) -> Shape
	func decode(buffer: BufferUInt8, to outData: inout DataView) throws
	func decodeInfo(buffer: BufferUInt8) throws -> (DataType, Shape)
	func encode(data: DataView, using stream: DeviceStream?, completion: EncodedHandler) throws
	func recode(buffer: BufferUInt8, using stream: DeviceStream?, completion: EncodedHandler) throws
}

public typealias EncodedHandler = (DataType, Shape, BufferUInt8) throws -> Void

// CodecType
public enum CodecType : String, EnumerableType {
	case data, audio, image, video
}

