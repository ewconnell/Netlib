//******************************************************************************
//  Created by Edward Connell on 2/2/17
//  Copyright © 2016 Connell Research. All rights reserved.
//
import Cuda
import CudaKernels
import Foundation

public final class CudaPseudoRandomGeneratorState: CudaRandomGeneratorState, Logging {
	// initializers
	public init(log: Log?, using stream: DeviceStream, count: Int,
	            seed: UInt?, name: String) throws {
		assert(stream is CudaStream)
		self.stream = stream
		self.count  = count
		self.name   = name
		currentLog  = log

		// select the device
		try stream.device.select()

		// seed
		let generatorSeed: UInt64 = seed != nil ?
			UInt64(seed!) : UInt64(Date().timeIntervalSince1970)

		// create the initial state
		var temp: UnsafeMutableRawPointer?
		try cudaCheck(status: cudaCreateRandomGeneratorState(
			&temp,
			generatorSeed,
			count,
			(stream as! CudaStream).handle))

		handle = temp!

		// log
		trackingId = objectTracker.register(type: self)
		if willLog(level: .diagnostic) {
			diagnostic("\(createString) \(name)", categories: .dataAlloc)
		}
	}

	// deinit
	deinit {
		do {
			// select the device
			try stream.device.select()

			// make sure pending queued commands complete before destroying the queue
			try stream.blockCallerUntilComplete()

			// free the memory
			cudaFree(handle);

			// remove from object tracking
			objectTracker.remove(trackingId: trackingId)
		} catch {
			writeLog(String(describing: error))
		}

		if willLog(level: .diagnostic) == true {
			diagnostic("\(releaseString) \(name)", categories: .dataAlloc)
		}
	}

	//----------------------------------------------------------------------------
	// properties
	public private(set) var trackingId = 0
	public private(set) var stream: DeviceStream
	public let count: Int
	public private(set) var handle: UnsafeMutableRawPointer

	// logging
	public var name: String
	public var logLevel = LogLevel.error
	public var nestingLevel = 0
	public weak var currentLog: Log?
}

public protocol CudaRandomGeneratorState : RandomGeneratorState {
	var handle: UnsafeMutableRawPointer { get }
}

