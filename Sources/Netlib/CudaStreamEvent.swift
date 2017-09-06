//******************************************************************************
//  Created by Edward Connell on 11/18/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
import Cuda

public final class CudaStreamEvent : StreamEvent {

	// initializers
	public required init(options: StreamEventOptions = []) throws {
		// the default is a non host blocking, non timing, non inter process event
		var flags: Int32 = cudaEventDisableTiming
		if !options.contains(.timing)      { flags &= ~cudaEventDisableTiming }
		if options.contains(.interprocess) { flags |= cudaEventInterprocess |
			                                            cudaEventDisableTiming }
		if options.contains(.hostSync)     { flags |= cudaEventBlockingSync }

		var temp: cudaEvent_t?
		try cudaCheck(status: cudaEventCreateWithFlags(&temp, UInt32(flags)))
		handle = temp!
		trackingId = objectTracker.register(type: self)
	}

	deinit {
		_ = cudaEventDestroy(handle)
		objectTracker.remove(trackingId: trackingId)
	}

	//----------------------------------------------------------------------------
	// properties
	public private (set) var trackingId = 0
	public var handle: cudaEvent_t

	//----------------------------------------------------------------------------
	// occurred
	public var occurred: Bool {
		return cudaEventQuery(handle) == cudaSuccess
	}
}
