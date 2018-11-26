//******************************************************************************
//  Created by Edward Connell on 3/21/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
import Cuda
import Dispatch

public class CudaDevice : ComputeDevice {
	// initializers
	public init(log: Log?, service: CudaComputeService, id: Int) throws {
		self.service = service
		self.id = id
		currentLog = log

		// get device properties
		var tempProps = cudaDeviceProp()
		try cudaCheck(status: cudaGetDeviceProperties(&tempProps, 0))
		props = tempProps
		let nameCapacity = MemoryLayout.size(ofValue: tempProps.name)

		name = withUnsafePointer(to: &tempProps.name) {
			$0.withMemoryRebound(to: UInt8.self, capacity: nameCapacity) {
				String(cString: $0)
			}
		}
		initAttributes()

		// this is a singleton
		trackingId = objectTracker.register(type: self, info: "device: \(id)")
		objectTracker.markStatic(trackingId: trackingId)

	}
	deinit { objectTracker.remove(trackingId: trackingId) }

	//----------------------------------------------------------------------------
	// properties
	public private (set) var trackingId = 0
	public private (set) weak var service: ComputeService!
	public var attributes = [String : String]()
	public var availableMemory: Int = 0
	public var maxThreadsPerBlock: Int { return Int(props.maxThreadsPerBlock) }
	public let name: String
	public let id: Int
	public let props: cudaDeviceProp
	public func supports(dataType: DataType) -> Bool { return true }
	private var streamId = AtomicCounter(value: -1)

	public var usesUnifiedAddressing: Bool {
		return props.concurrentManagedAccess == 1 &&
			props.pageableMemoryAccess == 1
	}

	// logging
	public var logLevel = LogLevel.error
	public var nestingLevel = 0
	public weak var currentLog: Log?

	//----------------------------------------------------------------------------
	// initAttributes
	private func initAttributes() {
		attributes = [
			"name"               : self.name,
			"compute capability" : "\(props.major).\(props.minor)",
			"global memory"      : "\(props.totalGlobalMem / (1024 * 1024)) MB",
			"multiprocessors"    : "\(props.multiProcessorCount)",
			"unified addressing" : "\(usesUnifiedAddressing ? "yes" : "no")",
		]
	}

	//----------------------------------------------------------------------------
	// createArray
//	private var totalMem = 0
	public func createArray(count: Int) throws -> DeviceArray {
//		totalMem += count
//		print("Create gpu array: \(count)  total: \(Float(totalMem) / Float(1.MB))MB")
		return try CudaDeviceArray(log: currentLog, device: self, count: count)
	}
	
	//----------------------------------------------------------------------------
	// createStream
	public func createStream(label: String) throws -> DeviceStream {
		return try CudaStream(log: currentLog, device: self, id: streamId.increment(),
			                    label: "\(label)_d\(id)")
	}

	//----------------------------------------------------------------------------
	// select
	public func select() throws {
		try cudaCheck(status: cudaSetDevice(Int32(id)))
	}
} // CudaDevice
