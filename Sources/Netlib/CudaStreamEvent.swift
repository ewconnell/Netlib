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
