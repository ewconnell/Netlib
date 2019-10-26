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

