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


