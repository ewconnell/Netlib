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
import Cuda

public class CudaDeviceArray : DeviceArray, ObjectTracking {
	// initializers
	public init(log: Log?, device: CudaDevice, count: Int) throws {
		self.device = device
		self.count  = count
		currentLog  = log

		try device.select()
		var devicePointer: UnsafeMutableRawPointer?
		try cudaCheck(status: cudaMalloc(&devicePointer, count))
		data = devicePointer!
		streamWaitEvent = try CudaStreamEvent()
		trackingId = objectTracker.register(type: self, info: "count: \(count)")
	}
	
	deinit {
		do {
			try device.select()
			try cudaCheck(status: cudaFree(data))
			objectTracker.remove(trackingId: trackingId)
		} catch {
			writeLog("\(releaseString) CudaDeviceArray(\(trackingId)) \(String(describing: error))")
		}
	}
	
	//----------------------------------------------------------------------------
	// properties
	public private(set) var trackingId = 0
	public let device: ComputeDevice
	public let data: UnsafeMutableRawPointer
	public let count: Int
	public var version = 0
	private var streamWaitEvent: CudaStreamEvent

	// logging
	public var logLevel = LogLevel.error
	public var nestingLevel = 0
	public weak var currentLog: Log?

	//----------------------------------------------------------------------------
	// zero
	public func zero(using stream: DeviceStream?) throws {
		let cudaStream = (stream as! CudaStream).handle
		try cudaCheck(status: cudaMemsetAsync(data, Int32(0), count, cudaStream))
	}

	//----------------------------------------------------------------------------
	// synchronous copy(to buffer
	public func copy(to buffer: MutableBufferUInt8, using stream: DeviceStream) throws {
		try copyAsync(to: buffer, using: stream)
		try stream.blockCallerUntilComplete()
	}

	//----------------------------------------------------------------------------
	// copyAsync(from deviceArray
	public func copyAsync(from deviceArray: DeviceArray,
	                      using stream: DeviceStream) throws {
		assert(deviceArray is CudaDeviceArray)
		let stream = stream as! CudaStream
		try stream.device.select()

		// copy
		try cudaCheck(status: cudaMemcpyAsync(
			data, UnsafeRawPointer(deviceArray.data),
			count, cudaMemcpyDeviceToDevice, stream.handle))
	}
	
	//----------------------------------------------------------------------------
	// copyAsync(from buffer
	public func copyAsync(from buffer: BufferUInt8, using stream: DeviceStream) throws {
		let stream = stream as! CudaStream
		try stream.device.select()

		try cudaCheck(status: cudaMemcpyAsync(
			data,	UnsafeRawPointer(buffer.baseAddress!),
			count, cudaMemcpyHostToDevice, stream.handle))
	}
	
	//----------------------------------------------------------------------------
	// copyAsync(to buffer
	public func copyAsync(to buffer: MutableBufferUInt8, using stream: DeviceStream) throws {
		let stream = stream as! CudaStream
		try stream.device.select()

		try cudaCheck(status: cudaMemcpyAsync(
			UnsafeMutableRawPointer(buffer.baseAddress!),
			UnsafeRawPointer(data), count,
			cudaMemcpyDeviceToHost, stream.handle))
	}
}
