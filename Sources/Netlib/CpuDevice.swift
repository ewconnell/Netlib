//******************************************************************************
//  Created by Edward Connell on 3/21/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
import Foundation
import Dispatch

public class CpuDevice : ComputeDevice {
	// initializers
	public init(log: Log?, service: CpuComputeService, deviceId: Int) {
		self.currentLog = log
		self.id         = deviceId
		self.service    = service
		self.path       = "\(service.name).\(deviceId)"

		// this is a singleton
		trackingId = objectTracker.register(type: self)
		objectTracker.markStatic(trackingId: trackingId)
	}
	deinit { objectTracker.remove(trackingId: trackingId) }

	//----------------------------------------------------------------------------
	// properties
	public private(set) var trackingId = 0
	public var attributes = [String : String]()
	public var availableMemory: Int = 0
	public var maxThreadsPerBlock: Int { return 1 /*this should be number of cores*/ }
	public let name: String = "CPU"
	public weak var service: ComputeService!
	public let id: Int
	public let path: String
	public let usesUnifiedAddressing = true
	private var streamId = AtomicCounter()

	// logging
	public var logLevel = LogLevel.error
	public var nestingLevel = 0
	public weak var currentLog: Log?

	//----------------------------------------------------------------------------

	public func select() throws { }

	public func supports(dataType: DataType) -> Bool { return true }
	
	//-------------------------------------
	// createArray
	//	This creates memory on the device
	public func createArray(count: Int) throws -> DeviceArray {
		return CpuDeviceArray(log: currentLog, device: self, count: count)
	}

	//-------------------------------------
	// createStream
	public func createStream(label: String) throws -> DeviceStream {
		return try CpuStream(log: currentLog, device: self,
		                     id: streamId.increment(), label: label)
	}
} // CpuDevice


