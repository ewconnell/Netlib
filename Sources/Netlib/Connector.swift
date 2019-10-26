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
// A connector can be input, output, or act as both when used on a container

public final class Connector : ModelObjectBase {
	// initializers
	public convenience init(name: String) {
		self.init()
		self.name = name
	}

	public convenience init(name: String, connection: Connection,
	                        minConnections: Int = 1, maxConnections: Int = 1) {
		self.init()
		self.name = name
		self.minConnections = minConnections
		self.maxConnections = maxConnections
		connections.append(connection); onSet("connections")
		connection.name = "\(connections.count)"
	}

	public convenience init(name: String, minConnections: Int, maxConnections: Int) {
		self.init()
		self.name = name
		self.minConnections = minConnections
		self.maxConnections = maxConnections
	}

	//----------------------------------------------------------------------------
	// properties
	// connections are data flow relationships defined by the user
	// they can be explicit, or symbolic in the case of connecting containers
	public var connections = [Connection]()        { didSet{onSet("connections")}}

	// containers may store input references of internal elements to be connected
	// to external elements during setup. These will either be forwarded to an
	// outer container or resolved when a container is connected to an actual
	// computational element
	public var inputReferences: [Connector]?

	// by default container outputs that are not connected to a contained element
	// are redirected to a container input creating a 'pass through'.
	// For example: container.labels --> container.labelsInput
	public var redirect: Connector?

	// dependents are all of the actual computational dependents
	// which include both explicit and implied connections. This list should
	// not contain any containers, but only actual computational elements
	public var dependents = [Connector]()

	// the element that owns this connector
	public var element: ModelElement { return parent as! ModelElement }

	// the number of connections allowed is controlled by the derived
	// class and may be dynamic depending on system resources
	public var minConnections: Int = 0
	public var maxConnections: Int = 1
	public var willAcceptConnection: Bool {
		get { return dependents.count < maxConnections }
	}

	// this is a collection of output data items for this connector
	// there may be one per device
	public var items = [ConnectorItem]()

	//----------------------------------------------------------------------------
	// addAccessors
	public override func addAccessors() {
		super.addAccessors()
		addAccessor(name: "connections",
		            get: { [unowned self] in self.connections },
		            set: { [unowned self] in self.connections = $0 })
	}
}

//==============================================================================
// ConnectorItem
final public class ConnectorItem {
	// initializer
	public init(using stream: DeviceStream) {
		self.stream = stream
	}

	//------------------------------------
	// properties

	// the stream to use with this item
	public var stream: DeviceStream

	// data presented from a forward pass
	// possibly ground truth data if labelsInput is connected to the element
	// or it may be output computed from a probability set, like from Softmax
	public var data = DataView()

	// optional data computed during a forward pass
	// if backData == nil, then 'data' is used during a backward pass
	// if backData != nil, then 'backData' is used during a backward pass
	public var backData: DataView?

	// gradient for backward pass
	public var grad: DataView?

	// corresponding data containers (just from the database for now)
//	public var annotations: [DataContainer?]?
}

//==============================================================================
// Connection
//   This is an individual reference to a specific element.output
public final class Connection : ModelObjectBase {
	// initializers
	public convenience init(input: String) {
		self.init()
		self.input = input; onSet("input")
	}

	//----------------------------------------------------------------------------
	// properties
	public var input = ""	                              { didSet{onSet("input")} }

	public var otherElementName: String {
		get {	return input.components(separatedBy: ".")[0] }
	}
	public var otherConnectorName : String {
		get {
			let path = input.components(separatedBy: ".")
			return path.count > 1 ? path[1] : "data"
		}
	}

	//----------------------------------------------------------------------------
	// addAccessors
	public override func addAccessors() {
		super.addAccessors()
		addAccessor(name: "input",
		            get: { [unowned self] in self.input },
		            set: { [unowned self] in self.input = $0 })
	}
}
