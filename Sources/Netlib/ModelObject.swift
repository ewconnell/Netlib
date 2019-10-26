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

//==============================================================================
// ModelObject
public protocol ModelObject : ObjectTracking, Properties {
	// properties
	var streams: [DeviceStream]? { get set }

	// functions
	func setup(taskGroup: TaskGroup) throws
}

//==============================================================================
// ModelObjectBase
open class ModelObjectBase : PropertiesBase, ModelObject {
	//----------------------------------------------------------------------------
	// properties
	public var computeServiceName: String? { didSet{onSet("computeServiceName")} }
	public var deviceIds: [Int]?           { didSet{onSet("deviceIds")} }
	public var streams: [DeviceStream]?

	// local
	public var createTaskGroup = false
	private let exclude = Set<String>(arrayLiteral: "items", "templates")

	//----------------------------------------------------------------------------
	// addAccessors
	public override func addAccessors() {
		super.addAccessors()
		addAccessor(name: "computeServiceName",
			get: { [unowned self] in self.computeServiceName },
			set: { [unowned self] in self.computeServiceName = $0 })
		addAccessor(name: "deviceIds",
			get: { [unowned self] in self.deviceIds },
			set: { [unowned self] in self.deviceIds = $0 })
	}

	//----------------------------------------------------------------------------
	// setup
	//	calls setup on all dynamic properties
	public func setup(taskGroup: TaskGroup) throws {
		for prop in properties.values where !prop.isExplicitlySetup {
			try prop.setup(taskGroup: taskGroup)
		}
	}

	//------------------------------------
	// find - finds a descendant by name
	public func find<T: ModelObject>(type: T.Type, name: String) -> T? {
		var result: T?
		enumerateObjects {
			if $0.namespaceName == name {
				if let object = $0 as? T {
					// terminate search
					result = object
					return false
				} else {
					writeLog("find - type mismatch \($0.namePath) is " +
						"type: \(Swift.type(of: $0)) expected: \(T.self)")
				}
			}

			// continue search
			return true
		}

		return result
	}
}

