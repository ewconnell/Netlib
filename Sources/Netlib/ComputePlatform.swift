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
//  ComputePlatform
//		services
//			ComputeService
//				devices
//					ComputeDevice
//						DeviceStream
//						DeviceArray
import Foundation

//==============================================================================
// ComputePlatform
//  Platform initialization is one time and lazy
final public class ComputePlatform : ModelObjectBase {
	//----------------------------------------------------------------------------
	// properties
	public var defaultDeviceCount = 1            { didSet{onSet("defaultDeviceCount")} }
	public var devicePriority: [Int]?            { didSet{onSet("devicePriority")} }
	public var servicePriority = ["cuda", "cpu"] { didSet{onSet("servicePriority")} }

	// local
	private var initialized = false
	private var _defaultDevice: ComputeDevice!
	public var defaultDevice: ComputeDevice {
		get { initialize(); return _defaultDevice }
	}

	// cpu, cuda, etc...
	private var _services = [String : ComputeService]()
	private var services: [String : ComputeService] {
		get { initialize(); return _services }
		set { initialize(); _services = newValue }
	}

	//-----------------------------------
	// plugIns
	public static var plugInBundles: [Bundle] = {
		var bundles = [Bundle]()
		if let dir = Bundle.main.builtInPlugInsPath {
			let paths = Bundle.paths(forResourcesOfType: "bundle", inDirectory: dir)
			for path in paths {
				bundles.append(Bundle(url: URL(fileURLWithPath: path))!)
			}
		}
		return bundles
	}()

	//----------------------------------------------------------------------------
	// addAccessors
	public override func addAccessors() {
		super.addAccessors()
		addAccessor(name: "defaultDeviceCount",
		            get: { [unowned self] in self.defaultDeviceCount },
		            set: { [unowned self] in self.defaultDeviceCount = $0 })
		addAccessor(name: "devicePriority",
		            get: { [unowned self] in self.devicePriority },
		            set: { [unowned self] in self.devicePriority = $0 })
		addAccessor(name: "servicePriority",
		            get: { [unowned self] in self.servicePriority },
		            set: { [unowned self] in self.servicePriority = $0 })
	}

	//----------------------------------------------------------------------------
	// copy
	public override func copy(from other: Properties) {
		super.copy(from: other)
		let other = other as! ComputePlatform
		initialized    = other.initialized
		_defaultDevice = other._defaultDevice
		_services      = other._services
	}

	//----------------------------------------------------------------------------
	// initialize
	private func initialize() {
		// only do it once since this is an expensive shared object
		guard !initialized else { return }
		initialized = true

		do {
			// add cpu service by default
			try add(service: CpuComputeService(log: currentLog))
			#if os(Linux)
				try add(service: CudaComputeService(log: currentLog))
			#endif
			
			for bundle in ComputePlatform.plugInBundles {
				try bundle.loadAndReturnError()
//			var unloadBundle = false

				if let serviceType = bundle.principalClass as? ComputeService.Type {
					// create the service
					let service = try serviceType.init(log: currentLog)

					if willLog(level: .diagnostic) {
						diagnostic("Loaded compute service '\(service.name)'." +
							" Device count = \(service.devices.count)", categories: .setup)
					}

					if service.devices.count > 0 {
						// add plugin service
						add(service: service)
					} else {
						if willLog(level: .warning) {
							writeLog("Compute service '\(service.name)' successfully loaded, " +
								"but reported devices = 0, so service is unavailable",
								level: .warning)
						}
//					unloadBundle = true
					}
				}
				// TODO: we should call bundle unload here if there were no devices
				// however simply calling bundle.load() then bundle.unload() making no
				// references to objects inside, later causes an exception in the code.
				// Very strange
//			if unloadBundle { bundle.unload() }
			}

			// try to exact match the service request
			let requestedDevice = devicePriority?[0] ?? 0
			for serviceName in servicePriority where _defaultDevice == nil {
				_defaultDevice = requestDevice(serviceName: serviceName,
					deviceId: requestedDevice,
					allowSubstitute: false)
			}

			// if the search failed, then allow substitutes
			if _defaultDevice == nil {
				let services = servicePriority + ["cpu"]
				for serviceName in services where _defaultDevice == nil {
					_defaultDevice = requestDevice(serviceName: serviceName, deviceId: 0,
						allowSubstitute: false)
				}
			}

			// we have to find something
			assert(_defaultDevice != nil)
			if willLog(level: .status) {
				writeLog("default device: \(defaultDevice.name)" +
					"   id: \(defaultDevice.service.name).\(defaultDevice.id)",
					level: .status)
			}
		} catch {
			writeLog(String(describing: error))
		}
	}

	//----------------------------------------------------------------------------
	// add(service
	public func add(service: ComputeService) {
		service.id = services.count
		services[service.name] = service
	}

	//----------------------------------------------------------------------------
	// requestStreams
	//	This will try to match the requested service and device ids returning
	// substitutes if needed.
	//
	// If no service name is specified, then the default is used.
	// If no ids are specified, then one stream per defaultDeviceCount is returned
	public func requestStreams(label: String,
	                           serviceName: String? = nil,
	                           deviceIds: [Int]? = nil) throws -> [DeviceStream]
	{
		let serviceName = serviceName ?? defaultDevice.service.name
		let maxDeviceCount = min(defaultDeviceCount, defaultDevice.service.devices.count)
		let ids = deviceIds ?? [Int](0..<maxDeviceCount)
		
		return try ids.map {
			let device = requestDevice(serviceName: serviceName,
			                           deviceId: $0, allowSubstitute: true)!
			return try device.createStream(label: label)
		}
	}
	
	//----------------------------------------------------------------------------
	// requestDevices
	public func requestDevices(serviceName: String?, deviceIds: [Int]) -> [ComputeDevice] {
		// if no serviceName is specified then return the default
		let serviceName = serviceName ?? defaultDevice.service.name
		return deviceIds.map {
			requestDevice(serviceName: serviceName,
			              deviceId: $0, allowSubstitute: true)!
		}
	}

	//----------------------------------------------------------------------------
	// requestDevice
	//		This tries to satisfy the device requested, but if not available will
	// return a suitable alternative. In the case of an invalid string, an
	// error will be reported, but no exception will be thrown
	public func requestDevice(serviceName: String, deviceId: Int,
	                          allowSubstitute: Bool) -> ComputeDevice?
	{
		if let service = services[serviceName] {
			if deviceId < service.devices.count {
				return service.devices[deviceId]
			} else if allowSubstitute {
				return service.devices[deviceId % service.devices.count]
			} else {
				return nil
			}
		} else if allowSubstitute {
			let service = services[defaultDevice.service.name]!
			return service.devices[deviceId % service.devices.count]
		} else {
			return nil
		}
	}
} // ComputePlatform

