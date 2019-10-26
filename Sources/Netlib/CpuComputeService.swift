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
public class CpuComputeService : ComputeService {
	// initializers
	public required init(log: Log?) throws	{
		currentLog = log

		// this is a singleton
		trackingId = objectTracker.register(type: self)
		objectTracker.markStatic(trackingId: trackingId)

		devices.append(CpuDevice(log: log, service: self, deviceId: 0))
	}
	deinit { objectTracker.remove(trackingId: trackingId) }

	//----------------------------------------------------------------------------
	// properties
	public private(set) var trackingId = 0
	public let name = "cpu"
	public var id = 0
	public var devices = [ComputeDevice]()

	// logging
	public var logLevel = LogLevel.error
	public var nestingLevel = 0
	public weak var currentLog: Log?
}




