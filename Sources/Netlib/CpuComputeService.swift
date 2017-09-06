//******************************************************************************
//  Created by Edward Connell on 4/4/16
//  Copyright © 2016 Connell Research. All rights reserved.
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




