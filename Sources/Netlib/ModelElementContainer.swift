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
// ModelElementContainer
public protocol ModelElementContainer : ModelElement, DefaultPropertiesContainer {
	// properties
	var data: String? { get set }
	var items: ElementNamespace { get set }
	var labels: String? { get set }
	var parameters: [String : String]? { get set }
	var storage: Uri? { get set }
	var templates: [String : ModelElement]? { get set }

	// functions
	func connectElements() throws
	func connect(connector: Connector, toContainerInput: String) throws
	func enumerateElements(handler: (ModelElement) throws -> Bool) throws -> Bool
	func filter(_ isIncluded: (ModelElement) throws -> Bool) throws -> [ModelElement]
	func findTemplate(name templateName: String) -> ModelElement?
	func resolve(connectorName: String, inputPath: inout String?, impliedPath: String,
	             listName: String, list: inout [String : Connector]) throws
}

public typealias ElementNamespace = OrderedNamespace<ModelElementBase>

//==============================================================================
// ModelElementContainerBase
open class ModelElementContainerBase : ModelElementBase, ModelElementContainer {
	// Properties
	public override var isContainer: Bool { return true }

	// ModelElementContainer
	public var data: String?                       { didSet{onSet("data")} }
	public var items = ElementNamespace()          { didSet{onSet("items")} }
	public var labels: String?                     { didSet{onSet("labels")} }
	public var parameters: [String : String]?      { didSet{onSet("parameters")}}
	public var storage: Uri?                       { didSet{onSet("storage")}}
	public var templates: [String : ModelElement]? { didSet{onSet("templates")} }

	// DefaultPropertiesContainer
	public var defaults: [Default]? {didSet{onDefaultsChanged(); onSet("defaults")}}
	public var defaultTypeIndex = [String : [String : ModelDefault]]()
	public func onDefaultsChanged() { rebuildDefaultTypeIndex() }
	public var defaultValues: [String : String]? {
		didSet { onDefaultValuesChanged(oldValue: oldValue); onSet("defaultValues") }
	}

	//----------------------------------------------------------------------------
	// addAccessors
	public override func addAccessors() {
		super.addAccessors()
		addAccessor(name: "data", lookup: .noLookup,
		            get: { [unowned self] in self.data },
		            set: { [unowned self] in self.data = $0 })
		addAccessor(name: "defaults",
		            get: { [unowned self] in self.defaults },
		            set: { [unowned self] in self.defaults = $0 })
		addAccessor(name: "defaultValues", lookup: .noLookup,
		            get: { [unowned self] in self.defaultValues },
		            set: { [unowned self] in self.defaultValues = $0 })
		addAccessor(name: "items",
		            get: { [unowned self] in self.items },
		            set: { [unowned self] in self.items = $0 })
		addAccessor(name: "labels", lookup: .noLookup,
		            get: { [unowned self] in self.labels },
		            set: { [unowned self] in self.labels = $0 })
		addAccessor(name: "parameters", lookup: .noLookup,
		            get: { [unowned self] in self.parameters },
		            set: { [unowned self] in self.parameters = $0 })
		addAccessor(name: "storage", lookup: .noLookup,
		            get: { [unowned self] in self.storage },
		            set: { [unowned self] in self.storage = $0 })
		addAccessor(name: "templates",
		            get: { [unowned self] in self.templates },
		            set: { [unowned self] in self.templates = $0 })
	}

	//----------------------------------------------------------------------------
	// findStorageLocation
	public override func findStorageLocation() throws -> URL? {
		return try storage?.getURL() ?? parent?.findStorageLocation()
	}
	
	//----------------------------------------------------------------------------
	// forward
	// This should only be called for the root invocation case
	// otherwise elements across containers should be directly connected.
	// Sub containers fall away during evaluation
	public override func forward(mode: EvaluationMode,
	                             selection: Selection) throws -> FwdProgress {
		// check if we already have the selection available
		guard selection != currentFwdSelection else { return currentFwdProgress }
		currentFwdSelection = selection

		if willLog(level: .diagnostic) {
			diagnostic("\(namespaceName)::forward container", categories: .evaluate)
		}

		// forward all dependents so data is available on outputs
		var ep = 0.0
		for output in outputs.values {
			for dependent in output.dependents {
				// TODO: should the currentEpoch be the same for all dependents?
				ep = try dependent.element.forward(mode: mode, selection: selection).epoch
			}
		}
		currentFwdProgress = FwdProgress(epoch: ep)

		return currentFwdProgress
	}

	//----------------------------------------------------------------------------
	// backward
	// This should only be called in the root invocation case
	// otherwise elements across containers should be directly connected
	// so sub containers fall away during evaluation
	public override func backward(solver: ModelSolver, selection: Selection) throws {
		// check if we already have already performed backward for selection
		guard selection != currentBackSelection else { return }
		currentBackSelection = selection

		if willLog(level: .diagnostic) {
			diagnostic("\(namespaceName)::backward container", categories: .evaluate)
		}

		// backward all dependents
		for output in outputs.values {
			for dependent in output.dependents {
				try dependent.element.backward(solver: solver, selection: selection)
			}
		}
	}

	//----------------------------------------------------------------------------
	// computeLoss
	public override func computeLoss() throws -> Double {
		// get cumulative loss of dependents
		var cumulativeLoss = 0.0

		for output in outputs.values {
			for dependent in output.dependents {
				cumulativeLoss += try dependent.element.computeLoss()
			}
		}
		return cumulativeLoss
	}

	//----------------------------------------------------------------------------
	// copyOutput
	//	This utility function is used to retrieve selected data from the specified
	// connector. If no itemIndex is specified, then the data items for the
	// specified dependent are concatenated to return a combined result.
	//   The outData buffer will be reused if it is of sufficient size,
	// otherwise a new one will be allocated.
	public override func copyOutput(connectorName: String,
	                                to outData: inout DataView,
	                                using stream: DeviceStream? = nil,
	                                dependentIndex: Int = 0,
	                                itemIndex: Int? = nil) throws {
		// get the dependent output connector
		assert(outputs[connectorName] != nil)
		let connector = outputs[connectorName]!
		guard connector.dependents.count > 0 else {
			writeLog("copyOutput failed because " +
				"'\(namespaceName).\(connectorName)' is not connected")
			throw ModelError.evaluationFailed
		}

		try connector.dependents[dependentIndex].element.copyOutput(
			connectorName: connectorName, to: &outData, using: stream,
			dependentIndex: dependentIndex, itemIndex: itemIndex)
	}

	//----------------------------------------------------------------------------
	// setup
	public override func setup(taskGroup: TaskGroup) throws {
		assert(model != nil)

		// apply the template if one is specified and
		// prevent base setup for applying it again
		if let element = try findTemplate() {
			try applyTemplate(templateElement: element)
			doApplyTemplate = false
		}
		
		// set the implied input name for this element if not defined
		for (i, element) in items.enumerated() {
			if element.input == nil {
				if i < items.count - 1 {
					element.input = items[i + 1].namespaceName
				} else {
					element.input = ".input"
				}
				element.properties["input"]!.isGenerated = true
			}
		}

		// templates is first because the collection may be
		// used by other collections on this same object
		// This is to ensure proper template chaining
		let templatesProp = properties["templates"]!
		try templatesProp.setup(taskGroup: taskGroup)
		templatesProp.isExplicitlySetup = true
		
		// generate implied inputs and outputs
		// use a temp to optionally add both connectors at once to avoid
		// multiple initialization
		var tempOutputs = outputs
		try resolve(connectorName: "data", inputPath: &data,
		            impliedPath: items.isEmpty ? ".input" : items[0].namespaceName,
		            listName: "outputs", list: &tempOutputs)
		
		try resolve(connectorName: "labels", inputPath: &labels,
		            impliedPath: ".labelsInput", listName: "outputs",
		            list: &tempOutputs)
		outputs = tempOutputs
		
		var tempInputs = inputs
		try resolve(connectorName: "input", inputPath: &input, impliedPath: ".input",
		            listName: "inputs", list: &tempInputs)
		
		try resolve(connectorName: "labelsInput", inputPath: &labelsInput,
		            impliedPath: ".labelsInput", listName: "inputs",
		            list: &tempInputs)
		inputs = tempInputs

		// setup base
		try super.setup(taskGroup: taskGroup)
	}

	//----------------------------------------------------------------------------
	// resolve(connectorName
	//  Containers provide optional convenience connector properties so that
	// full input/output connection specification through collections are not
	// necessary in the majority of cases.
	//  This function resolves a connector name/path property and tries to
	// connect them. If the connector property has no value, then a passthrough
	// is set up.
	//
	public func resolve(connectorName: String, inputPath: inout String?,
	                    impliedPath: String, listName: String,
	                    list: inout [String : Connector]) throws {
		// resolve input
		if inputPath == nil {
			inputPath = impliedPath
			properties[connectorName]!.isGenerated = true
		}
		let path = inputPath!

		if let connector = list[connectorName] {
			if connector.connections.isEmpty {
				properties[listName]!.isGenerated = true
			} else {
				// skip duplicates in the case of multiple setup
				for connection in connector.connections {
					if connection.input == path { return }
				}
			}

			// add the connection
			connector.connections.append(Connection(input: path))
			connector.connections.last!.name = "\(connector.connections.count)"
		} else {
			// add connector and connection
			list[connectorName] =
				Connector(name: connectorName, connection: Connection(input: path))
			properties[listName]!.isGenerated = true
		}
	}

	//----------------------------------------------------------------------------
	// connectElements
	public func connectElements() throws {
		if willLog(level: .diagnostic) {
			diagnostic("", categories: .connections)
			diagnostic("container connections", categories: .connections,
			           trailing: "-", minCount: 80)
			diagnostic("\(namePath)", categories: .connections)
		}

		// a dependency path is maintained for cycle detection
		var dependencyPath = DependencyPath()

		// in case of multiple setup
		for input in inputs.values { input.inputReferences?.removeAll() }

		// connect outputs
		for output in outputs.values {
			try connectOutput(output: output, connections: output.connections,
				dependencyPath: &dependencyPath)
		}
	}

	//----------------------------------------------------------------------------
	// connectOutput
	//  finds and connects the dependencies of the specified output connector
	// and connection list. The connections may not belong to output in the case
	// of connection redirection in a container.
	private func connectOutput(output: Connector, redirected: Bool = false,
	                           connections: [Connection],
	                           dependencyPath: inout DependencyPath) throws {
		// in case of multiple setup
		output.dependents.removeAll()

		// validate the connection
		for connection in connections {
			if connection.otherElementName.isEmpty {
				// no element name implies a connection to the container
				try connect(connector: output,
					toContainerInput: connection.otherConnectorName)

			} else {
				// get names
				let otherElementName = connection.otherElementName
				let otherConnectorName = connection.otherConnectorName

				// locate element and connector
				guard let otherElement = items[otherElementName] else	{
					writeLog("\(namespaceName).\(output.namespaceName) --> '\(otherElementName)'" +
						" element not found. Connection failed")
					throw ModelError.setupFailed
				}
				guard let otherConnector = otherElement.outputs[otherConnectorName] else {
					writeLog("\(namespaceName).\(output.namespaceName) --> '\(connection.input)'" +
						" connector '\(otherConnectorName)' not found on" +
						" element '\(otherElementName)'. Connection failed")
					throw ModelError.setupFailed
				}

				// check for redirection of other container output to input
				if let input = otherConnector.redirect {
					// in a redirect, use the input's connections as if they
					// belong to the output
					try connectOutput(output: output, redirected: true,
						connections: input.connections,
						dependencyPath: &dependencyPath)
				} else {
					if otherElement.isContainer {
						// the other connector is a container connector, so add all of
						// it's contained dependents to this output's dependents
						for dependent in otherConnector.dependents {
							output.dependents.append(dependent)
						}
					} else {
						// the other connector is an individual element. Store it as an
						// internal dependency for fix up in the outer scope
						output.dependents.append(otherConnector)
					}
					if willLog(level: .diagnostic) {
						var message = "[.\(output.namespaceName)] --> " +
							"\(otherElement.namespaceName).\(otherConnectorName)"
						if redirected { message += " (redirected)"}
						diagnostic(message, categories: .connections, indent: 1)
					}

					// connect element to it's inputs. Consumed elements will be removed
					// from the model set. The dependency path is used for cycle detection
					if dependencyPath.isNotConnected(name: otherElement.namespaceName) {
						try otherElement.connectInputs(dependencyPath: &dependencyPath)
					}
				}
			}
		}
	}

	//----------------------------------------------------------------------------
	// connectInputs
	public override func connectInputs(dependencyPath: inout DependencyPath) throws {
		guard let container = self.container else { return }

		// do cycle check. If this element's name is in the path, then we've
		// made a cycle which will cause an infinite evaluation loop
		if dependencyPath.contains(name: namespaceName) {
			writeLog("Cycle detected \(String(dependencyPath))")
			throw ModelError.setupFailed
		} else {
			// add this elements name to the path to track where we've been
			dependencyPath.push(name: namespaceName)
		}

		// connect each input
		for input in inputs.values {
			// in case of multiple setup
			input.dependents.removeAll()

			for connection in input.connections {
				let otherElementName = connection.otherElementName
				let otherConnectorName = connection.otherConnectorName

				if otherElementName.isEmpty {
					//------------------------------
					// connection to container
					if willLog(level: .diagnostic) {
						diagnostic("• \(namespaceName).\(input.namespaceName) --> " +
							"[\(container.namespaceName).\(otherConnectorName)]",
							categories: .connections)
					}
					try container.connect(connector: input,
						toContainerInput: otherConnectorName)
				} else {
					//------------------------------
					// connection to sibling element
					guard let otherElement =
					container.items[connection.otherElementName] else {
						writeLog("\(namespaceName).\(input.namespaceName) --> " +
							"'\(connection.otherElementName)'" +
							" element not found. Connection failed")
						throw ModelError.setupFailed
					}

					// connect each input ref to the other element's output
					// through container abstraction, this can possibly lead to a
					// many to many relationship
					if let refs = input.inputReferences {
						if willLog(level: .diagnostic) {
							diagnostic("• \(namespaceName).\(input.namespaceName) --> " +
								"\(otherElementName).\(otherConnectorName)",
								categories: .connections)
						}

						for ref in refs {
							try otherElement.connect(outputName: otherConnectorName, to: ref)
							if willLog(level: .diagnostic) {
								for dep in ref.dependents {
									if otherElement !== dep.element {
										diagnostic("direct connect: \(namespaceName).\(ref.element.namespaceName).\(ref.namespaceName) --> " +
											"\(otherElementName).\(dep.element.namespaceName).\(otherConnectorName)",
											categories: .connections, indent: 2)
									} else {
										diagnostic("direct connect: \(namespaceName).\(ref.element.namespaceName).\(ref.namespaceName) --> " +
											"\(otherElementName).\(otherConnectorName)",
											categories: .connections, indent: 2)
									}
								}
							}
						}
					}

					// propagate down the tree if not already connected
					if dependencyPath.isNotConnected(name: otherElement.namespaceName) {
						try otherElement.connectInputs(dependencyPath: &dependencyPath)
						dependencyPath.pop()
					}
				}
			}
		}
	}

	//----------------------------------------------------------------------------
	// connect output
	//   A container output may connect internally to a single or multiple
	// elements.
	public override func connect(outputName: String, to input: Connector) throws {
		guard let output = outputs[outputName] else {
			writeLog("Connector '\(outputName)' not found on " +
				"'\(namespaceName)'. Connection failed")
			throw ModelError.setupFailed
		}

		for dependent in output.dependents {
			if input.willAcceptConnection {
				// containers cannot be directly connected
				assert(!dependent.element.isContainer,
					"containers cannot be directly connected")
				input.dependents.append(dependent)
			} else {
				writeLog("\(input.element.namespaceName).\(input.namespaceName) connection limit is " +
					"\(input.maxConnections). Connection failed")
				throw ModelError.setupFailed
			}
		}
	}

	//----------------------------------------------------------------------------
	// connect toContainerInput
	//   Elements that connect to a container input, have their connections
	// stored in an input reference list for later fix up when this container
	// is connected in it's parent scope
	public func connect(connector: Connector, toContainerInput: String) throws {
		guard let input = inputs[toContainerInput] else {
			writeLog("Connector '\(toContainerInput)' not found on " +
				"'\(namespaceName)'. Connection failed")
			throw ModelError.setupFailed
		}

		// save connections from internal element
		if connector.element.isContainer {
			if let connectorRefs = connector.inputReferences {
				assert(connectorRefs.count != 0)
				// lazy create ref list
				if input.inputReferences == nil { input.inputReferences = [] }

				// add references to all of the container elements that are
				// connected to "connector" onto the input list, thus forwarding
				// the request for a later fix up
				input.inputReferences!.append(contentsOf: connectorRefs)

				if willLog(level: .diagnostic) {
					diagnostic("[.\(connector.namespaceName)] --> [.\(input.namespaceName)]",
						categories: .connections, indent: 1)

					let list = connectorRefs.map { "\($0.element.namespaceName).\($0.namespaceName)" }
					diagnostic("input references delegated \(list)",
						categories: .connections, indent: 2)
				}
			} else {
				// redirect the output to the input
				connector.redirect = input

				if willLog(level: .diagnostic) && connector.namespaceName != input.namespaceName {
					diagnostic("[.\(connector.namespaceName)] --> [.\(input.namespaceName)] " +
						"(pass through)", categories: .connections, indent: 1)
				}
			}
		} else {
			// lazy create ref list and add TODO: not thread safe, but do we care?
			// setup is single threaded for now
			if input.inputReferences == nil { input.inputReferences = [] }
			input.inputReferences!.append(connector)
		}
	}

	//----------------------------------------------------------------------------
	// findTemplate
	//  returns a reference to a sibling or ancestor element by name
	//  looking in the templates and items collections
	public func findTemplate(name templateName: String) -> ModelElement? {
		// search containers toward root
		var currentContainer: ModelElementContainer! = self
		while currentContainer != nil {
			if let element = currentContainer.templates?[templateName] ??
				currentContainer.items[templateName] {
				return element
			}
			currentContainer = currentContainer.container
		}
		return nil
	}

	//----------------------------------------------------------------------------
	// filter - filters siblings and dependents
	public func filter(_ isIncluded: (ModelElement) throws -> Bool) throws -> [ModelElement] {
		var result = [ModelElement]()
		try enumerateElements {
			if try isIncluded($0) { result.append($0) }
			return true
		}
		return result
	}

	//----------------------------------------------------------------------------
	// enumerateElements
	//	 Enumerates all descendent elements
	//
	// return
	//		false terminates the enumeration process
	//
	@discardableResult
	public func enumerateElements(handler: (ModelElement) throws -> Bool) throws -> Bool {
		// walk all elements at this level
		for element in items {
			if !(try handler(element)) { return false }
		}

		// traverse child containers
		for element in items {
			if let container = element as? ModelElementContainer {
				if !(try container.enumerateElements(handler: handler)) { return false }
			}
		}
		return true
	}

	//----------------------------------------------------------------------------
	// lookupDefault
	public override func lookupDefault(typePath: String, propPath: String) -> ModelDefault? {
		return lookup(typePath: typePath, propPath: propPath) ??
			parent?.lookupDefault(typePath: typePath, propPath: propPath)
	}
}


