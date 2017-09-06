//******************************************************************************
//  Created by Edward Connell on 6/2/16
//  Copyright © 2016 Connell Research. All rights reserved.
//

//------------------------------------------------------------------------------
// ModelDataContainer
public protocol ModelDataContainer : ModelObject, BinarySerializable {
	var codec: Codec? { get set }
	var codecType: CodecType? { get set }
	var extent: [Int]? { get set }
	var dataLayout: DataLayout { get set }
	var dataType: DataType { get set }
	var colMajor: Bool { get set }
	var shape: Shape { get }
	var sourceIndex: Int { get set }
	var transform: Transform? { get set }
	var uri: Uri? { get set }
	var uriString: String? { get set }
	var value: [Double]? { get set }

	func decode(data: inout DataView) throws
	func encode(completion: EncodedHandler) throws
	func encodedShape() throws -> Shape
}

//------------------------------------------------------------------------------
// ModelLabeledContainer
public protocol ModelLabeledContainer : ModelDataContainer {
	var label     : Label? { get set }
	var labelValue: [Double]? { get set }

	func decode(data: inout DataView, label: inout DataView?) throws
	func verifyEqual(to other: ModelLabeledContainer) throws
}

//------------------------------------------------------------------------------
// NetLabel
public protocol NetLabel : ModelDataContainer {
	var rangeOffset: [Int]?	{ get set }
	var rangeExtent: [Int]? { get set }
}

//==============================================================================
// DataContainerBase
//    This is a container for media types such as Image, Audio, etc...
open class DataContainerBase : ModelObjectBase, ModelDataContainer {
	// initializers
	public required init() { super.init() }

	//----------------------------------------------------------------------------
	// BinarySerializable init
	//	This points to the data source
	public required init(from buffer: BufferUInt8, next: inout BufferUInt8) throws {
		super.init()
		try deserialize(from: buffer, next: &next)
	}

	// this optionally retains a copy of the data in case the source is transient
	public init(from buffer: BufferUInt8, next: inout BufferUInt8, retain: Bool) throws {
		super.init()
		try deserialize(from: buffer, next: &next)

		if retain {
			// copy what the encodedBuffer now points to
			encodedData = [UInt8](encodedBuffer)

			// point to the retained copy
			encodedBuffer = encodedData!.withUnsafeBufferPointer { $0 }
		}
	}

	//----------------------------------------------------------------------------
	// properties
	public var codec: Codec?                  { didSet{onSet("codec")} }
	public var codecType: CodecType?          { didSet{onSet("codecType")} }
	public var dataLayout = DataLayout.nchw   { didSet{onSet("dataLayout")} }
	public var dataType = DataType.real8U     { didSet{onSet("dataType")} }
	public var extent: [Int]?                 { didSet{onSet("extent")} }
	public var colMajor = false               { didSet{onSet("colMajor")} }
	public var transform: Transform?					{ didSet{onSet("transform")} }
	public var uri: Uri?                      { didSet{onSet("uri")} }
	public var uriString: String?             { didSet{onSet("uriString")} }
	public var value: [Double]?               { didSet{onSet("value")} }
	
	// source index must be assigned by the owning DataSource object
	public var sourceIndex = -1

	// shape
	private var _shape: Shape!
	public var shape: Shape {
		get { return _shape }
		set { _shape = newValue }
	}
	
	// local
	public var data: DataView?
	fileprivate var encodedBuffer: BufferUInt8!
	fileprivate var encodedData: [UInt8]?
	fileprivate var codecTypeName: String?

	//----------------------------------------------------------------------------
	// addAccessors
	public override func addAccessors() {
		super.addAccessors()
		addAccessor(name: "codec",
		            get: { [unowned self] in self.codec },
		            set: { [unowned self] in self.codec = $0 })
		addAccessor(name: "codecType",
		            get: { [unowned self] in self.codecType },
		            set: { [unowned self] in self.codecType = $0 })
		addAccessor(name: "colMajor",
		            get: { [unowned self] in self.colMajor },
		            set: { [unowned self] in self.colMajor = $0 })
		addAccessor(name: "dataLayout",
			          get: { [unowned self] in self.dataLayout },
			          set: { [unowned self] in self.dataLayout = $0 })
		addAccessor(name: "dataType",
		            get: { [unowned self] in self.dataType },
		            set: { [unowned self] in self.dataType = $0 })
		addAccessor(name: "extent",
		            get: { [unowned self] in self.extent },
		            set: { [unowned self] in self.extent = $0 })
		addAccessor(name: "transform",
		            get: { [unowned self] in self.transform },
		            set: { [unowned self] in self.transform = $0 })
		addAccessor(name: "uri",
		            get: { [unowned self] in self.uri },
		            set: { [unowned self] in self.uri = $0 })
		addAccessor(name: "uriString",
		            get: { [unowned self] in self.uriString },
		            set: { [unowned self] in self.uriString = $0 })
		addAccessor(name: "value",
		            get: { [unowned self] in self.value },
		            set: { [unowned self] in self.value = $0 })
	}

	//----------------------------------------------------------------------------
	// encodedShape
	public func encodedShape() throws -> Shape {
		return codec?.encodedShape(of: shape) ?? shape
	}
	
	//-----------------------------------
	// serialize
	public func serialize(to buffer: inout [UInt8]) throws {
		try encode { dataType, shape, encodedBytes in
			try serializeOptional(codec?.typeName, to: &buffer)
			try serializeOptional(codecType, to: &buffer)
			assert(sourceIndex >= 0)
			sourceIndex.serialize(to: &buffer)
			dataType.serialize(to: &buffer)
			shape.serialize(to: &buffer)
			encodedBytes.serialize(to: &buffer)

			// reflect the encoded shape on the container
			_shape = shape
		}
	}

	//-----------------------------------
	// deserialize
	private func deserialize(from buffer: BufferUInt8, next: inout BufferUInt8) throws {
		codecTypeName = try deserializeOptional(from: buffer, next: &next)
		codecType = try deserializeOptional(from: next, next: &next)
		sourceIndex = Int(from: next, next: &next)
		dataType = DataType(from: next, next: &next)
		shape = Shape(from: next, next: &next)
		encodedBuffer = UnsafeBufferPointer(from: next, next: &next)

		// resolve the codec
		try resolveCodec()
	}

	//----------------------------------------------------------------------------
	// copy
	public override func copy(from other: Properties) {
		let other = other as! DataContainerBase

		// first copy the environment and context for logging purposes
		namePath = other.namePath
		propPath = other.propPath
		typePath = other.typePath

		// copy dynamic properties
		super.copy(from: other)

		// copy type specific
		codecTypeName = other.codecTypeName
		data = other.data
		encodedBuffer = other.encodedBuffer
		encodedData = other.encodedData
		sourceIndex = other.sourceIndex
	}

	//----------------------------------------------------------------------------
	// setup
	public override func setup(taskGroup: TaskGroup) throws {
		try super.setup(taskGroup: taskGroup)

		// resolve the codec
		try resolveCodec()

		// convenience string
		if uri == nil, let string = uriString {
			uri = Uri(string: string)
			properties["uri"]!.isGenerated = true
		}

		// setup data
		if let data = self.data {
			dataType = data.dataType
			shape = data.shape

		} else if let value = self.value {
			shape = Shape(extent: extent ?? [1, 1, 1, value.count],
			              layout: dataLayout, colMajor: colMajor)

			// get the data and do type/arrangement conversion if necessary
			data = try DataView(from: DataView(array: value, shape: shape),
				                  asDataType: dataType)

		} else if let url = try uri?.getURL() {
			// load data if a url is specified
			encodedData = try [UInt8](contentsOf: url)
			encodedBuffer = encodedData!.withUnsafeBufferPointer { $0 }
			let info = try codec!.decodeInfo(buffer: encodedBuffer)
			dataType = info.0
			shape = info.1
		}
	}

	//----------------------------------------------------------------------------
	// resolveCodec
	private func resolveCodec() throws {
		if codec == nil {
			// check if a specific codec type name was specified
			if let type = codecTypeName {
				codec = (try Create(typeName: type) as! Codec)
			} else {
				guard let type = codecType else {
					writeLog("codec or codecType must be specified for the DataContainer")
					throw ModelError.setupFailed
				}
				codec = try Model.createCodec(for: type)
				codecTypeName = codec!.typeName
			}
			properties["codec"]!.isGenerated = true
		}
	}
	
	//----------------------------------------------------------------------------
	// encode
	//	This takes the 'data' in whatever format it is in, and appends it to
	// the output buffer in the format associated with the codec
	public func encode(completion: EncodedHandler) throws {
		assert(encodedBuffer != nil || data != nil)
		// transform
//		if let transform = self.transform {
//			// decode the data
//			
//		}

		// append encoded data
		if let data = self.data {
			try codec!.encode(data: data, using: nil, completion: completion)
		} else {
			try codec!.recode(buffer: encodedBuffer, using: nil, completion: completion)
		}
	}
	
	//----------------------------------------------------------------------------
	// decode
	//	This decodes the data and writes it to the dataView
	public func decode(data: inout DataView) throws {
		assert(codec != nil, "call setup")
		try codec!.decode(buffer: encodedBuffer, to: &data)
		dataType = data.dataType
		shape = data.shape
	}
} // DataContainerBase

//==============================================================================
// DataContainer
//    This is a container for media types such as Image, Audio, etc...
public final class DataContainer : DataContainerBase, ModelLabeledContainer, Copyable {
	// initializers
	public required init() { super.init() }

	public convenience init(data: DataView, label: Label? = nil) {
		self.init()
		self.data = data
		if let label = label { self.label = label }
	}

	public convenience init(contentsOf uri: Uri, codecType: CodecType? = nil,
	                        label: Label? = nil) {
		self.init()
		self.uri = uri
		if let codecType = codecType { self.codecType = codecType }
		if let label = label { self.label = label }
	}

	//----------------------------------------------------------------------------
	// BinarySerializable init
	public required init(from buffer: BufferUInt8, next: inout BufferUInt8) throws {
		try super.init(from: buffer, next: &next)
		try deserialize(from: next, next: &next)
	}

	// variation that allows for retention of the buffer data
	// for non-memory mapped providers
	public override init(from buffer: BufferUInt8, next: inout BufferUInt8,
	                     retain: Bool) throws {
		try super.init(from: buffer, next: &next, retain: retain)
		try deserialize(from: next, next: &next)
	}

	//-----------------------------------
	// deserialize
	private func deserialize(from buffer: BufferUInt8, next: inout BufferUInt8) throws {
		labelValue = try deserializeOptional(from: buffer, next: &next)
		label = try deserializeOptional(from: next, next: &next)
	}

	//-----------------------------------
	// serialize
	public override func serialize(to buffer: inout [UInt8]) throws {
		guard label == nil || labelValue == nil else {
			writeLog("DataContainer: only set label or labelValue, not both")
			throw ModelError.conversationFailed("")
		}

		// append container data
		try super.serialize(to: &buffer)
		try serializeOptional(labelValue, to: &buffer)
		try serializeOptional(label, to: &buffer)
	}

	//----------------------------------------------------------------------------
	// properties
	public var label: Label?                       { didSet{onSet("label")} }
	public var labelValue: [Double]?               { didSet{onSet("labelValue")} }

	// computed prop
	public var labelShape: Shape? {
		if let values = labelValue {
			return Shape(count: values.count)
		} else {
			return label?.shape
		}
	}

	//----------------------------------------------------------------------------
	// addAccessors
	public override func addAccessors() {
		super.addAccessors()
		addAccessor(name: "label",
		            get: { [unowned self] in self.label },
		            set: { [unowned self] in self.label = $0 })
		addAccessor(name: "labelValue",
		            get: { [unowned self] in self.labelValue },
		            set: { [unowned self] in self.labelValue = $0 })
	}

	//----------------------------------------------------------------------------
	// verifyEqual
	public func verifyEqual(to other: ModelLabeledContainer) throws {
		let other = other as! DataContainer
		assert(self.codecTypeName == other.codecTypeName)
		assert(self.codecType == other.codecType)
		assert(self.dataType == other.dataType)
		if self.labelValue != nil {
			assert(self.label == nil && other.label == nil)
			assert(self.labelValue! == other.labelValue!)
		} else if self.label != nil {

		}
		if self.encodedBuffer != nil {
			assert(Array(self.encodedBuffer!) == Array(other.encodedBuffer!))
		}

		var otherData = DataView(shape: other.shape, dataType: other.dataType)
		var otherLabel: DataView? = DataView(shape: other.labelShape!, dataType: other.dataType)
		try other.decode(data: &otherData, label: &otherLabel)

		let originalDataArray = try Array(self.data!.roReal8U())
		let otherDataArray = try Array(otherData.roReal8U())
		assert(originalDataArray == otherDataArray)

		print(self.data!.format(columnWidth: 3, precision: 1, maxItems: 1, highlightThreshold: 0))
		print(otherData.format(columnWidth: 3, precision: 1, maxItems: 1, highlightThreshold: 0))
	}

	//----------------------------------------------------------------------------
	// decode
	//	This decodes the data and writes it to the dataView
	public func decode(data: inout DataView, label outLabel: inout DataView?) throws {
		try super.decode(data: &data)
		if outLabel != nil {
			if let values = labelValue {
				for i in 0..<values.count {
					try outLabel!.set(value: values[i], at: [i])
				}
			} else if let label = self.label {
				try label.decode(data: &outLabel!)

			} else {
				// we shouldn't get here
				assertionFailure("Label view supplied without container label." +
					" All containers in a set should either have a label or not")
			}
		}
	}
}

//==============================================================================
// Label
public final class Label : DataContainerBase, NetLabel {
	// initializers
	public required init() {
		super.init()
	}

	//----------------------------------------------------------------------------
	// BinarySerializable init
	public required init(from buffer: BufferUInt8, next: inout BufferUInt8) throws {
		try super.init(from: buffer, next: &next)
		try deserialize(from: next, next: &next)
	}
	
	//-----------------------------------
	// deserialize
	private func deserialize(from buffer: BufferUInt8, next: inout BufferUInt8) throws {
		rangeOffset = try deserializeOptional(from: buffer, next: &next)
		rangeExtent = try deserializeOptional(from: next, next: &next)
	}
	
	//-----------------------------------
	// serialize
	public override func serialize(to buffer: inout [UInt8]) throws {
		try super.serialize(to: &buffer)
		try serializeOptional(rangeOffset, to: &buffer)
		try serializeOptional(rangeExtent, to: &buffer)
	}

	//----------------------------------------------------------------------------
	// properties
	public var rangeOffset: [Int]?	              { didSet{onSet("rangeOffset")} }
	public var rangeExtent: [Int]?                { didSet{onSet("rangeExtent")} }

	//----------------------------------------------------------------------------
	// addAccessors
	public override func addAccessors() {
		super.addAccessors()
		addAccessor(name: "rangeOffset",
		            get: { [unowned self] in self.rangeOffset },
		            set: { [unowned self] in self.rangeOffset = $0 })
		addAccessor(name: "rangeExtent",
		            get: { [unowned self] in self.rangeExtent },
		            set: { [unowned self] in self.rangeExtent = $0 })
	}
}
