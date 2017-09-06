//******************************************************************************
//  Created by Edward Connell on 3/21/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
import Foundation
import Dispatch

public class DataArray : BinarySerializable, ObjectTracking, Logging {
	// initializers
	public convenience init() {
		self.init(log: nil, dataType: .real32F, elementCount: 0)
	}

	// All initializers retain the data except this one
	// which creates a read only reference to avoid unnecessary copying from
	// a database
	public init(log: Log?, dataType: DataType, readOnlyReferenceTo buffer: BufferUInt8) {
		isReadOnlyReference = true
		currentLog    = log
		self.dataType = dataType
		elementCount  = buffer.count
		masterVersion = 0
		hostVersion   = 0

		// we won't ever actually mutate in this case
		hostBuffer = MutableBufferUInt8(
			start: UnsafeMutablePointer(mutating: buffer.baseAddress),
			count: buffer.count)

		register()
	}

	//----------------------------------------
	// create new space
	public init(log: Log?, dataType: DataType, elementCount: Int, name: String? = nil) {
		isReadOnlyReference = false
		currentLog        = log
		self.dataType     = dataType
		self.elementCount = elementCount
		_name = name
		register()
	}

	//----------------------------------------
	// copy from buffer
	public init(dataType: DataType, buffer: BufferUInt8) {
		isReadOnlyReference = false
		self.dataType = dataType
		self.elementCount = buffer.count
		_ = try! rwReal8U().initialize(from: buffer)
		assert(hostVersion == 0 && masterVersion == 0)
		register()
	}

	//----------------------------------------
	// from BinarySerializable
	public required init(from buffer: BufferUInt8, next: inout BufferUInt8) {
		isReadOnlyReference = false
		dataType = DataType(from: buffer, next: &next)
		elementCount = Int(from: next, next: &next)
		_ = try! rwReal8U().initialize(from: buffer)
		assert(hostVersion == 0 && masterVersion == 0)
		register()
	}

	//----------------------------------------
	// init from url
	public init(dataType: DataType, contentsOf url: URL) throws {
		isReadOnlyReference = false
		self.dataType = dataType
		
		// load data and validate the count
		// TODO: load data directly into data array buffer instead of
		// using Data class
		let data = try Data(contentsOf: url)
		guard data.count % dataType.size == 0 else {
			throw ModelError.error(
				"DataArray uri data count must be a multiple of the specified dataType")
		}
		elementCount = data.count / dataType.size
		_ = try rwReal8U().initialize(from: data)
		assert(hostVersion == 0 && masterVersion == 0)
		register()
	}

	//----------------------------------------
	// init from other DataArray
	public init(withContentsOf other: DataArray,
	            using stream: DeviceStream? = nil) throws {
		// init
		isReadOnlyReference = other.isReadOnlyReference
		dataType            = other.dataType
		elementCount        = other.elementCount
		name                = other.name
		masterVersion       = 0
		hostVersion = masterVersion
		register()

		if willLog(level: .diagnostic) {
			let streamIdStr = stream == nil ? "nil" : "\(stream!.id)"
			diagnostic("\(createString) \(name)(\(trackingId)) init" +
				"\(setText(" copying ", color: .blue))" +
				"DataArray(\(other.trackingId)) elements: \(other.elementCount) " +
				"stream id(\(streamIdStr))", categories: [.dataAlloc, .dataCopy])
		}

		if isReadOnlyReference {
			// point to external data buffer, such as LMDB memory mapped data record
			assert(master == nil)
			hostBuffer = other.hostBuffer

		} else if let stream = stream {
			// get new array for the target stream's device location
			let arrayInfo = try getArray(for: stream)
			let array     = arrayInfo.array
			array.version = masterVersion

			if let otherMaster = other.master {
				// sync streams and copy
				try stream.sync(with: otherMaster.stream,
					              event: getSyncEvent(using: stream))
				try array.copyAsync(from: otherMaster.array, using: stream)

			} else {
				// uma to device
				try array.copyAsync(from: other.roReal8U(), using: stream)
			}

			// set the master
			master = arrayInfo

		} else {
			// get pointer to this array's umaBuffer
			let buffer = try rwReal8U()

			if let otherMaster = other.master {
				// synchronous device to umaArray
				try otherMaster.array.copy(to: buffer, using: otherMaster.stream)

			} else {
				// umaArray to umaArray
				_ = try buffer.initialize(from: other.roReal8U())
			}
		}
	}

	// object lifetime tracking for leak detection
	private func register() {
		trackingId = objectTracker.register(
			type: self, info: "elementCount: \(elementCount)")

		if elementCount > 0 && willLog(level: .diagnostic) {
			diagnostic("\(createString) \(name)(\(trackingId)) " +
				"elements: \(elementCount)", categories: .dataAlloc)
		}
	}

	deinit {
		do {
			// synchronize with all streams that have accessed these arrays
			// before freeing them
			for sid in 0..<deviceArrays.count {
				for devId in 0..<deviceArrays[sid].count {
					if let info = deviceArrays[sid][devId] {
						try info.stream.blockCallerUntilComplete()
					}
				}
			}
		} catch {
			writeLog(String(describing: error))
		}
		objectTracker.remove(trackingId: trackingId)

		if elementCount > 0 && willLog(level: .diagnostic) {
			diagnostic("\(releaseString) \(name)(\(trackingId)) " +
				"elements: \(elementCount)", categories: .dataAlloc)
		}
	}

	//----------------------------------------------------------------------------
	// properties
	public let accessQueue = DispatchQueue(label: "DataArray.accessQueue")
	public let elementCount: Int
	public var byteCount: Int { get { return dataType.size * elementCount } }
	public let dataType: DataType
	public var autoReleaseUmaBuffer = false
	public private(set) var trackingId = 0

	// constants
	public lazy var zero: AnyValue = { AnyValue(dataType: self.dataType, value: 0) }()
	public lazy var one: AnyValue = { AnyValue(dataType: self.dataType, value: 1) }()

	// logging
	public var logLevel = LogLevel.error
	public var nestingLevel = 0
	public weak var currentLog: Log?

	// name
	private var _name: String?
	public var name: String {
		get {
			return _name ?? "DataArray"
		}
		set {
			_name = newValue
		}
	}

	// testing
	public private(set) var lastAccessCopiedBuffer = false

	// local
	private let streamRequired = "stream is required for device data transfers"
	private let isReadOnlyReference: Bool
	private var hostArray = [UInt8]()
	private var hostBuffer: MutableBufferUInt8!
	private var hostVersion = -1

	// stream sync
	private var _streamSyncEvent: StreamEvent!
	private func getSyncEvent(using stream: DeviceStream) throws -> StreamEvent {
		if _streamSyncEvent == nil {
			_streamSyncEvent = try stream.createEvent(options: [])
		}
		return _streamSyncEvent
	}

	// this can either point to the hostArray or to the deviceArray
	// depending on the location of the master
	private var deviceDataPointer: UnsafeMutableRawPointer!

	// this is indexed by [service.id][device.id]
	// and contains a lazy allocated array on each device,
	// which is a replica of the current master
	private var deviceArrays = [[ArrayInfo?]]()

	public class ArrayInfo {
		public init(array: DeviceArray, stream: DeviceStream) {
			self.array = array
			self.stream = stream
		}

		public let array: DeviceArray
		// stream is tracked for synchronous cleanup (deinit) of the array
		public var stream: DeviceStream
	}

	// whenever a buffer write pointer is taken, the associated DeviceArray
	// becomes the master copy for replication. Synchronization across threads
	// is still required for taking multiple write pointers, however
	// this does automatically synchronize data migrations.
	// A value of nil means that the master is the umaBuffer
  public private(set) var master: ArrayInfo?

	// this is incremented each time a write pointer is taken
	// all replicated buffers will stay in sync with this version
	private var masterVersion = -1

	//----------------------------------------------------------------------------
	// willLog
	//  all normal objects require the environment to be defined, however
	// this is relaxed for Data View/Array
	public func willLog(level: LogLevel) -> Bool {
		guard let log = currentLog else { return false }
		return level <= log.logLevel || level <= logLevel
	}

	//----------------------------------------------------------------------------
	// BinarySerializable
	public func serialize(to buffer: inout [UInt8]) {
		dataType.serialize(to: &buffer)
		buffer.serialize(to: &buffer)
	}

	//----------------------------------------------------------------------------
	// ro
	public func roReal8U() throws -> BufferUInt8 {
		try migrate(readOnly: true)
		return BufferUInt8(hostBuffer)
	}
	
	public func ro(using stream: DeviceStream) throws -> UnsafeRawPointer {
		try migrate(readOnly: true, using: stream)
		return UnsafeRawPointer(deviceDataPointer)
	}
	
	//----------------------------------------------------------------------------
	// rw
	public func rwReal8U() throws -> MutableBufferUInt8 {
		assert(!isReadOnlyReference)
		try migrate(readOnly: false)
		return hostBuffer
	}

	public func rw(using stream: DeviceStream) throws -> UnsafeMutableRawPointer {
		assert(!isReadOnlyReference)
		try migrate(readOnly: false, using: stream)
		return deviceDataPointer
	}

	//----------------------------------------------------------------------------
	// migrate
	private func migrate(readOnly: Bool, using stream: DeviceStream? = nil) throws {
		// if the array is empty then there is nothing to do
		guard !isReadOnlyReference && elementCount > 0 else { return }
		let srcUsesUMA = master?.stream.device.usesUnifiedAddressing ?? true
		let dstUsesUMA = stream?.device.usesUnifiedAddressing ?? true

		// reset, this is to support automated tests
		lastAccessCopiedBuffer = false

		switch srcUsesUMA {
		case true where dstUsesUMA:
			try setDeviceDataPointerToHostBuffer(readOnly: readOnly)

		case false where dstUsesUMA:
			try device2host(readOnly: readOnly)

		case true where !dstUsesUMA:
			assert(stream != nil, streamRequired)
			try host2device(readOnly: readOnly, using: stream!)

		case false where !dstUsesUMA:
			assert(stream != nil, streamRequired)
			try device2device(readOnly: readOnly, using: stream!)

		// shouldn't be possible
		default: fatalError()
		}
	}

	//----------------------------------------------------------------------------
	// getArray
	//  this manages a dictionary of replicated device arrays indexed
	// by serviceId and deviceId. It will lazily create a device array if needed
	private func getArray(for stream: DeviceStream) throws -> ArrayInfo {
		let device = stream.device
		let serviceId = device.service.id
		
		// add the device array list if needed
		if deviceArrays.count <= serviceId {
			let addCount = max(serviceId + 1, 2) - deviceArrays.count
			for _ in 0..<addCount {	deviceArrays.append([ArrayInfo?]()) }
		}

		// create array list if needed
		if deviceArrays[serviceId].isEmpty {
			deviceArrays[serviceId] = [ArrayInfo?](repeating: nil,
				count: device.service.devices.count)
		}

		// return existing if found
		if let info = deviceArrays[serviceId][device.id] {
			// sync the requesting stream with the last stream that accessed it
			try stream.sync(with: info.stream, event: getSyncEvent(using: stream))

			// update the last stream used to access this array for sync purposes
			info.stream = stream
			return info

		} else {
			// create the device array
			if willLog(level: .diagnostic) {
				diagnostic("\(createString) \(name)(\(trackingId)) " +
					"allocating array on device(\(device.id)) elements: \(elementCount)",
					categories: .dataAlloc)
			}
			let array = try device.createArray(count: byteCount)
			array.version = -1
			let info = ArrayInfo(array: array, stream: stream)
			deviceArrays[serviceId][device.id] = info
			return info
		}
	}

	//----------------------------------------------------------------------------
	// createHostArray
	private func createHostArray() throws {
		if willLog(level: .diagnostic) {
			diagnostic("\(createString) \(name)(\(trackingId)) " +
				"host array  elements: \(elementCount)", categories: .dataAlloc)
		}
		hostArray = [UInt8](repeating: 0, count: byteCount)
		hostBuffer = hostArray.withUnsafeMutableBufferPointer { $0 }
		hostVersion = -1
	}

	//-----------------------------------
	// releaseHostArray
	private func releaseHostArray() {
		precondition(!isReadOnlyReference)
		if willLog(level: .diagnostic) {
			diagnostic(
				"\(releaseString) \(name) DataArray(\(trackingId)) host array " +
				"elements: \(elementCount)", categories: .dataAlloc)
		}
		hostArray = [UInt8]()
		hostBuffer = nil
	}

	//----------------------------------------------------------------------------
	// setDeviceDataPointerToHostBuffer
	private func setDeviceDataPointerToHostBuffer(readOnly: Bool) throws {
		assert(!isReadOnlyReference)
		// lazily create the uma buffer if needed
		if hostBuffer == nil { try createHostArray() }
		deviceDataPointer = UnsafeMutableRawPointer(hostBuffer.baseAddress!)
		if !readOnly { master = nil; masterVersion += 1 }
		hostVersion = masterVersion
	}

	//----------------------------------------------------------------------------
	// host2device
	private func host2device(readOnly: Bool, using stream: DeviceStream) throws {
		let arrayInfo = try getArray(for: stream)
		let array     = arrayInfo.array
		deviceDataPointer = array.data

		if hostBuffer == nil {
			// clear the device buffer and set it to be the new master
			try array.zero(using: stream)
			master = arrayInfo

		} else if array.version != masterVersion {
			// copy host data to device if it exists and is needed
			if willLog(level: .diagnostic) {
				diagnostic("\(copyString) \(name)(\(trackingId)) host" +
					"\(setText(" ---> ", color: .blue))" +
					"d\(stream.device.id)_s\(stream.id) elements: \(elementCount)",
					categories: .dataCopy)
			}

			try array.copyAsync(from: BufferUInt8(hostBuffer), using: stream)
			lastAccessCopiedBuffer = true

			if autoReleaseUmaBuffer && !isReadOnlyReference {
				// wait for the copy to complete, free the uma array,
				// and specify the device array as the new master
				try stream.blockCallerUntilComplete()
				releaseHostArray()
				master = arrayInfo
			}
		}

		// set version
		if !readOnly { master = arrayInfo; masterVersion += 1 }
		array.version = masterVersion
	}

	//----------------------------------------------------------------------------
	// device2host
	private func device2host(readOnly: Bool) throws {
		// master cannot be nil
		let master = self.master!
		assert(master.array.version == masterVersion)

		// lazily create the uma buffer if needed
		if hostBuffer == nil { try createHostArray() }
		deviceDataPointer = UnsafeMutableRawPointer(hostBuffer.baseAddress!)

		// copy if needed
		if hostVersion != masterVersion {
			if willLog(level: .diagnostic) {
				diagnostic("\(copyString) \(name)(\(trackingId)) " +
					"d\(master.stream.device.id)_s\(master.stream.id)" +
					"\(setText(" ---> ", color: .blue)) host" +
					" elements: \(elementCount)", categories: .dataCopy)
			}

			// synchronous copy
			try master.array.copy(to: hostBuffer, using: master.stream)
			lastAccessCopiedBuffer = true
		}

		// set version
		if !readOnly { self.master = nil; masterVersion += 1 }
		hostVersion = masterVersion
	}

	//----------------------------------------------------------------------------
	// device2device
	private func device2device(readOnly: Bool, using stream: DeviceStream) throws {
		// master cannot be nil
		let master = self.master!
		assert(master.array.version == masterVersion)

		// get array for stream's device and set deviceBuffer pointer
		let arrayInfo = try getArray(for: stream)
		let array     = arrayInfo.array
		deviceDataPointer = array.data

		// synchronize output stream with master stream
		try stream.sync(with: master.stream, event: getSyncEvent(using: stream))

		// copy only if versions do not match
		if array.version != masterVersion {
			// copy within same service
			if master.stream.device.service.id == stream.device.service.id {
				// copy cross device within the same service if needed
				if master.stream.device.id != stream.device.id {
					if willLog(level: .diagnostic) {
						diagnostic("\(copyString) \(name)(\(trackingId)) " +
							"device(\(master.stream.device.id))" +
							"\(setText(" ---> ", color: .blue))" +
							"device(\(stream.device.id)) elements: \(elementCount)",
							categories: .dataCopy)
					}
					try array.copyAsync(from: master.array, using: stream)
					lastAccessCopiedBuffer = true
				}

			} else {
				fatalError()
//				if willLog(level: .diagnostic) == true {
//					diagnostic("\(copyString) \(name)(\(trackingId)) cross service from " +
//						"device(\(master.stream.device.id))" +
//				    "\(setText(" ---> ", color: .blue))" +
//						"device(\(stream.device.id)) elementCount: \(elementCount)",
//						categories: .dataCopy)
//				}
//
//				// cross service non-uma migration
//				// copy data to uma buffer
//				if umaBuffer == nil { try createHostArray() }
//				try master.array.copy(to: umaBuffer, using: master.stream)
//
//				// copy data to destination device
//				try dest.array.copyAsync(from: BufferUInt8(umaBuffer), using: stream)
//
//				if autoReleaseUmaBuffer {
//					// wait for the copy to complete, free the uma array,
//					// and specify the device array as the new master
//					try stream.blockCallerUntilComplete()
//					releaseHostArray()
//					self.master = dest
//				}
//
//				lastAccessCopiedBuffer = true
			}
		}

		// set version
		if !readOnly { self.master = arrayInfo; masterVersion += 1 }
		self.master!.array.version = masterVersion
		array.version = masterVersion
	}

} // DataArray

