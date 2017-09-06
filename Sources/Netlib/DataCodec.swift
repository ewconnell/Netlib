//******************************************************************************
//  Created by Edward Connell on 10/18/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
import Zlib

final public class DataCodec : ModelObjectBase, Codec {
	public convenience init(compression: Int) {
		self.init()
		self.compression = compression
	}

	//----------------------------------------------------------------------------
	// properties
	public var compression = -1                   { didSet{onSet("compression")} }
	public var codecType: CodecType { return .data }

	//----------------------------------------------------------------------------
	// addAccessors
	public override func addAccessors() {
		super.addAccessors()
		addAccessor(name: "compression",
		            get: { [unowned self] in self.compression },
		            set: { [unowned self] in self.compression = $0 })
	}

	//----------------------------------------------------------------------------
	// encodedShape(of:
	public func encodedShape(of shape: Shape) -> Shape { return shape }
	
	//----------------------------------------------------------------------------
	// decode
	public func decode(buffer: BufferUInt8, to outData: inout DataView) throws {
		var next = BufferUInt8()
		let header = TensorHeader(from: buffer, next: &next)
		// read the zipped array count to advance next to point to zipped data
		_ = Int(from: next, next: &next)
		let unzipped = try unzip(buffer: next)
		
		let array = DataArray(log: currentLog, dataType: header.dataType,
			readOnlyReferenceTo: unzipped.withUnsafeBufferPointer {$0})

		let view = DataView(shape: header.shape, dataType: header.dataType,
		                    dataArray: array)
		
		// this may rearrange and do type conversion in the process if needed
		try cpuCopy(from: view, to: &outData)
	}
	
	//----------------------------------------------------------------------------
	// decodeInfo
	public func decodeInfo(buffer: BufferUInt8) throws -> (DataType, Shape) {
		var next = BufferUInt8()
		let header = TensorHeader(from: buffer, next: &next)
		return (header.dataType, header.shape)
	}
	
	//----------------------------------------------------------------------------
	// encode
	public func encode(data: DataView, using stream: DeviceStream? = nil,
	                   completion: EncodedHandler) throws {
		var output = [UInt8]()
		let shape = data.shape.elementCount == 1 ? Shape(count: 1) : data.shape
		let header = TensorHeader(dataType: data.dataType, shape: shape)
		header.serialize(to: &output)
		try zip(buffer: data.roReal8U(), compression: compression).serialize(to: &output)
		try completion(data.dataType, shape, output.withUnsafeBufferPointer { $0 })
	}
	
	//----------------------------------------------------------------------------
	// recode
	//	The only affect this really has is to possibly change
	// the compression level
	//
	public func recode(buffer: BufferUInt8, using stream: DeviceStream? = nil,
	                   completion: EncodedHandler) throws {
		
		let info = try decodeInfo(buffer: buffer)
		var temp = DataView(shape: info.1, dataType: info.0)
		try decode(buffer: buffer, to: &temp)
		try encode(data: temp, using: stream, completion: completion)
	}
} // DataCodec





