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
