//******************************************************************************
//  Created by Edward Connell on 3/21/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
public class CpuDeviceArray : DeviceArray {
	// initializers
	public init(log: Log?, device: ComputeDevice, count: Int) {
		currentLog  = log
		self.device = device
		self.count  = count

		data = UnsafeMutableRawPointer(bitPattern: 0)!
		trackingId = objectTracker.register(type: self)
	}
	deinit { objectTracker.remove(trackingId: trackingId) }

	//----------------------------------------------------------------------------
	// properties
	public private(set) var trackingId = 0
	public var device: ComputeDevice
	public var data: UnsafeMutableRawPointer
	public var count: Int
	public var version = 0

	// logging
	public var logLevel = LogLevel.error
	public var nestingLevel = 0
	public weak var currentLog: Log?

	//----------------------------------------------------------------------------
	// zero
	public func zero(using stream: DeviceStream?) throws {

	}

	// copyAsync(from deviceArray
	public func copyAsync(from other: DeviceArray, using stream: DeviceStream) throws {

	}

	// copyAsync(from buffer
	public func copyAsync(from buffer: BufferUInt8, using stream: DeviceStream) throws {

	}

	// copy(to buffer
	public func copy(to buffer: MutableBufferUInt8, using stream: DeviceStream) throws {

	}

	// copyAsync(to buffer
	public func copyAsync(to buffer: MutableBufferUInt8, using stream: DeviceStream) throws {

	}
}
