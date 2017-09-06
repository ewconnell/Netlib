//******************************************************************************
//  Created by Edward Connell on 5/9/16
//  Copyright © 2016 Connell Research. All rights reserved.
//

//==============================================================================
// ModelElement
public protocol ModelElement : ModelObject {
	// properties
	var container: ModelElementContainer? { get }
	var doSetupForward: Bool { get set }
	var doSetupBackward: Bool { get set }
	var input: String? { get set }
	var inputs: [String : Connector] { get set }
	var labelsInput: String? { get set }
	var learningError: Double { get set }
	var lossWeight: Double { get set }
	var outputLoss: Double { get set }
	var outputs: [String : Connector] { get set }
	var template: String? { get set }

	// functions
	func findTemplate() throws -> ModelElement?

	// forward
	func setupForward(mode: EvaluationMode, selection: Selection) throws
	func forward(mode: EvaluationMode, selection: Selection) throws -> FwdProgress

	// backward
	func setupBackward() throws
	func backward(solver: ModelSolver, selection: Selection) throws
	func computeLoss() throws -> Double

	// connect
	func connect(outputName: String, to input: Connector) throws
	func connectInputs(dependencyPath: inout DependencyPath) throws

	// join
	func copyOutput(connectorName: String, to outData: inout DataView,
	                using stream: DeviceStream?, dependentIndex: Int,
	                itemIndex: Int?) throws
}

//-------------------------------------
// FwdProgress
public struct FwdProgress {
	public init() { epoch = 0 }
	public init(epoch: Double) { self.epoch = epoch }
	public let epoch: Double
}

public enum EvaluationMode { case inference, training }

//--------------------------------------
// NanPropagation
public enum NanPropagation : String, EnumerableType {
	case propagate
	case noPropagate
}

//--------------------------------------
// MetricsElement
public protocol MetricsElement : ModelElement {
	func resetMetrics()
}

//--------------------------------------
// AccuracyElement
public protocol AccuracyElement : MetricsElement {
	var accuracy: Double { get }
	var error: Double { get }
	var evaluatedCount: Int { get }
}

//==============================================================================
// ModelElementBase
//  convenience base
open class ModelElementBase : ModelObjectBase, ModelElement {
	//----------------------------------------------------------------------------
	// properties
	public var input: String?                   { didSet{onSet("input")} }
	public var labelsInput: String?             { didSet{onSet("labelsInput")} }
	public var outputLoss = 0.0                 { didSet{onSet("outputLoss")} }
	public var learningError = 0.0              { didSet{onSet("learningError")} }
	public var lossWeight = 0.0                 { didSet{onSet("lossWeight")} }
	public var inputs = [String : Connector]()	{ didSet{onSet("inputs")} }
	public var outputs = [String : Connector]() { didSet{onSet("outputs")} }
	public var template: String?                { didSet{onSet("template")} }

	// non scripting
	public var doApplyTemplate = true
	public var doSetupForward = true
	public var doSetupBackward = true
	public var currentFwdSelection = Selection()
	public var currentFwdProgress = FwdProgress()
	public var currentBackSelection = Selection()
	public var container: ModelElementContainer? {
		get { return parent as? ModelElementContainer }
	}

	// abstracts
	public func onForward(mode: EvaluationMode, selection: Selection) throws { }
	public func onBackward(solver: ModelSolver) throws { }
	public func onComputeLoss() throws -> Double { return 0 }

	//----------------------------------------------------------------------------
	// addAccessors
	public override func addAccessors() {
		super.addAccessors()
		addAccessor(name: "input",
		            get: { [unowned self] in self.input },
		            set: { [unowned self] in self.input = $0 })
		addAccessor(name: "labelsInput",
		            get: { [unowned self] in self.labelsInput },
		            set: { [unowned self] in self.labelsInput = $0 })
		addAccessor(name: "outputLoss",
		            get: { [unowned self] in self.outputLoss },
		            set: { [unowned self] in self.outputLoss = $0 })
		addAccessor(name: "learningError",
		            get: { [unowned self] in self.learningError },
		            set: { [unowned self] in self.learningError = $0 })
		addAccessor(name: "lossWeight",
		            get: { [unowned self] in self.lossWeight },
		            set: { [unowned self] in self.lossWeight = $0 })
		addAccessor(name: "inputs",
		            get: { [unowned self] in self.inputs },
		            set: { [unowned self] in self.inputs = $0 })
		addAccessor(name: "outputs",
		            get: { [unowned self] in self.outputs },
		            set: { [unowned self] in self.outputs = $0 })
		addAccessor(name: "template",
		            get: { [unowned self] in self.template },
		            set: { [unowned self] in self.template = $0 })
	}

	//----------------------------------------------------------------------------
	// setup
	public override func setup(taskGroup: TaskGroup) throws {
		assert(model != nil)

		// copy the template properties if specified
		if doApplyTemplate, let element = try findTemplate() {
			try applyTemplate(templateElement: element)
		}

		// items is after the template has been applied, and is separate
		// because the collection may contain templated items
		// used by other collections on this same object
		// This is to ensure proper template chaining
		if let itemsProp = properties["items"] {
			try itemsProp.setup(taskGroup: taskGroup)
			itemsProp.isExplicitlySetup = true
		}
		
		// setup all the properties
		try super.setup(taskGroup: taskGroup)

		// setup forward/backward anytime the network configuration changes
		doSetupForward = true
		doSetupBackward = true
	}

	//----------------------------------------------------------------------------
	// setupForward
	public func setupForward(mode: EvaluationMode, selection: Selection) throws {
		// reset
		doSetupForward = false
		currentFwdSelection = Selection()
	}

	//----------------------------------------------------------------------------
	// forward
	public func forward(mode: EvaluationMode, selection: Selection) throws -> FwdProgress {
		// check if we already have the selection available
		guard selection != currentFwdSelection else { return currentFwdProgress }
		currentFwdSelection = selection

		// forward all dependents so data is available on inputs
		var ep = 0.0
		for input in inputs.values {
			for dependent in input.dependents {
				// TODO: should the currentEpoch be the same for all dependents?
				ep = try dependent.element.forward(mode: mode, selection: selection).epoch
			}
		}
		currentFwdProgress = FwdProgress(epoch: ep)

		// setup the element's forward pass
		if doSetupForward {	try setupForward(mode: mode, selection: selection) }

		// ask the element to do the forward
		if willLog(level: .diagnostic) {
			diagnostic("\(namespaceName).forward", categories: .evaluate)
		}
		try onForward(mode: mode, selection: selection)

		return currentFwdProgress
	}

	//----------------------------------------------------------------------------
	// setupBackward
	public func setupBackward() throws {
		doSetupBackward = false

		for input in inputs.values {
			for dependent in input.dependents {
				try dependent.element.setupBackward()
			}
		}
	}

	//----------------------------------------------------------------------------
	// backward
	public func backward(solver: ModelSolver, selection: Selection) throws {
		// check if we already have already performed backward for selection
		guard selection != currentBackSelection else { return }
		currentBackSelection = selection

		if doSetupBackward { try setupBackward() }

		// the derived class will compute the diffs and place the result
		// on the dependents outputs
		if willLog(level: .diagnostic) {
			diagnostic("\(namespaceName).backward", categories: .evaluate)
		}
		try onBackward(solver: solver)

		// backward to all dependents so the diffs are available
		// on the dependent's outputs
		for input in inputs.values {
			for dependent in input.dependents {
				try dependent.element.backward(solver: solver, selection: selection)
			}
		}
	}

	//----------------------------------------------------------------------------
	// computeLoss
	public func computeLoss() throws -> Double {
		// element loss plus dependent losses
		var dependentLoss = 0.0
		for input in inputs.values {
			for dependent in input.dependents {
				dependentLoss += try dependent.element.computeLoss()
			}
		}

		// only compute this element's loss if the weight is greater than 0
		if lossWeight > 0 { outputLoss = try onComputeLoss() }

		return outputLoss + dependentLoss
	}

	//----------------------------------------------------------------------------
	// connect
	public func connect(outputName: String, to input: Connector) throws {
		guard let output = outputs[outputName] else {
			writeLog("Connector '\(outputName)' not found on " +
				"'\(namespaceName)'. Connection failed")
			throw ModelError.setupFailed
		}
		input.dependents.append(output)
	}

	//----------------------------------------------------------------------------
	// connectInputs
	public func connectInputs(dependencyPath: inout DependencyPath) throws {
		guard let container = self.container else { return }

		// do cycle check. If this element's name is in the path, then we've
		// made a cycle which will cause an infinite evaluation loop
		if dependencyPath.contains(name: namespaceName) {
			throw ModelError.error("Cycle detected \(String(dependencyPath))")
		} else {
			// add this elements name to the path to track where we've been
			dependencyPath.push(name: namespaceName)
		}

		// connect each input
		for input in inputs.values {
			// use immutable original input connection list for resolution by using
			// .array because an individual connection may cause multiple connections
			// to be added when connecting to a function that encapsulates a fan in
			for connection in input.connections {
				let otherElementName = connection.otherElementName
				let otherConnectorName = connection.otherConnectorName

				if otherElementName.isEmpty {
					//------------------------------
					// connection to container
					try container.connect(connector: input,
						toContainerInput: otherConnectorName)

					if willLog(level: .diagnostic) {
						diagnostic("• \(namespaceName).\(input.namespaceName) --> [.\(otherConnectorName)]",
							categories: .connections)
					}
				} else {
					//------------------------------
					// connection to sibling element
					guard let element =	container.items[otherElementName] else {
						writeLog("\(namespaceName).\(input.namespaceName) --> '\(otherElementName)'" +
							" element not found. Connection failed")
						throw ModelError.setupFailed
					}

					// connect and remove the input from the roots list
					try element.connect(outputName: otherConnectorName, to: input)

					if willLog(level: .diagnostic) {
						diagnostic("• \(namespaceName).\(input.namespaceName) --> \(otherElementName)" +
							".\(otherConnectorName)", categories: .connections)

						if element.isContainer {
							for dep in input.dependents {
								diagnostic("direct connect: \(namespaceName).\(input.namespaceName) --> " +
									"\(otherElementName).\(dep.element.namespaceName).\(dep.namespaceName)",
									categories: .connections, indent: 2)
							}
						}
					}

					// propagate down the tree if not already connected
					if dependencyPath.isNotConnected(name: element.namespaceName) {
						try element.connectInputs(dependencyPath: &dependencyPath)
						dependencyPath.pop()
					}
				}
			}
		}
	}

	//----------------------------------------------------------------------------
	// copyOutput
	//	This utility function is used to retrieve selected data from the specified
	// connector. If no itemIndex is specified, then the data items for the
	// specified dependent are concatenated to return a combined result.
	//   The outData buffer will be reused if it is of sufficient size,
	// otherwise a new one will be allocated.
	public func copyOutput(connectorName: String, to outData: inout DataView,
	                       using stream: DeviceStream? = nil,
	                       dependentIndex: Int = 0, itemIndex: Int? = nil) throws {
		assert(outputs[connectorName] != nil)
		let connector = outputs[connectorName]!

		// if there are no outputs available
		if connector.items.count == 0 {
			outData = DataView()
			
		} else {
			// allocate the buffer if needed
			let joinedExtent = getJoinedDataExtent(for: connector.items)
			if outData.extent != joinedExtent {
				let shape = Shape(extent: joinedExtent,
					                layout: connector.items[0].data.shape.layout)
				outData = DataView(shape: shape, dataType: connector.items[0].data.dataType)
			}

			// join
			try joinDataItems(items: connector.items, using: stream, to: &outData)
		}
	}


	//----------------------------------------------------------------------------
	// getJoinedDataExtent
	public func getJoinedDataExtent(for items: [ConnectorItem],
	                                joinBackData: Bool = false) -> [Int] {
		if items.count == 0 {
			return []
		} else if items.count == 1 {
			return joinBackData ? items[0].backData?.shape.extent ??
				items[0].data.shape.extent : items[0].data.shape.extent
		} else {
			var count = 0
			for item in items {
				count += joinBackData ? item.backData?.extent[0] ??
					item.data.extent[0] : item.data.extent[0]
			}

			let item0Data = joinBackData ? items[0].backData ??
				items[0].data : items[0].data

			return [count] + item0Data.extent.suffix(from: 1)
		}
	}

	//----------------------------------------------------------------------------
	// joinDataItems
	//	This joins the data buffers in the items collection
	public func joinDataItems(items: [ConnectorItem],
	                          using stream: DeviceStream?,
	                          to outData: inout DataView,
	                          joinBackData: Bool = false) throws
	{
		var index = [Int](repeating: 0, count: outData.rank)

		for item in items {
			// create a writable reference output view for the item
			let itemExtent = joinBackData ? item.backData?.shape.extent ??
				item.data.shape.extent : item.data.shape.extent
			var outView = try outData.referenceView(offset: index, extent: itemExtent, using: stream)

			// copy data
			try item.stream.copy(from: joinBackData ? item.backData! : item.data,
			                     to: &outView, normalizeInts: false)

			// advance to next position
			index[0] += itemExtent[0]
		}

		// if there is no caller stream then sync the item streams
		if stream == nil {
			for item in items {
				try item.stream.blockCallerUntilComplete()
			}
		}
	}

	//----------------------------------------------------------------------------
	// findTemplate
	//	This locates the element specified by the optional template property
	public func findTemplate() throws -> ModelElement? {
		guard let elementName = template else { return nil	}
		guard let element = container?.findTemplate(name: elementName) else {
			writeLog("template: \(elementName) not found")
			throw ModelError.setupFailed
		}

		// make sure this object and the template are the same types
		guard element.typeName == typeName else {
			writeLog("TypeMismatch - template source '\(element.namespaceName)' " +
				"is type '\(element.typeName)'. Element '\(namespaceName)'" +
				" is type '\(self.typeName)'")
			throw ModelError.setupFailed
		}
		return element
	}

	//----------------------------------------------------------------------------
	// applyTemplate
	public func applyTemplate(templateElement: ModelElement) throws {
		diagnostic("copy template values", categories: .setup,
		           trailing: "-", minCount: 60)

		// copy any properties from the template not set on this object
		// and not part of the exclusion set
		let exclusionSet = Set<String>(
			["environment", "name", "storage", "template", "templateUri", TypeNameKey])
		
		// update default or previously templated values
		for (key, prop) in properties
			where !exclusionSet.contains(key) && prop.isDefault || prop.isTemplated
		{
			let templateProp = templateElement.properties[key]!
			if !templateProp.isDefault && !templateProp.isGenerated {
				// log
				if willLog(level: .diagnostic) {
					diagnostic("\(namespaceName).\(prop.name) = " +
						"\(templateElement.namespaceName).\(prop.name)",
						categories: .setup, indent: 1)
				}
				
				// copy
				prop.copy(other: templateProp)
				prop.isTemplated = true
			}
		}
	}
}

//==============================================================================
// DependencyPath
//  This is used internally during connectElements to track connection
// dependency paths for detecting cycles and independent roots
public struct DependencyPath {
	var path = [String]()
	var pathSet = Set<String>()
	var connected = Set<String>()
	mutating func push(name: String) {
		path.append(name)
		pathSet.insert(name)
		connected.insert(name)
	}
	mutating func pop() {	pathSet.remove(path.removeLast())	}
	func contains(name: String) -> Bool { return pathSet.contains(name) }
	func isNotConnected(name: String) -> Bool { return !connected.contains(name) }
}

extension String {
	public init(_ dependencyPath: DependencyPath) {
		let path = dependencyPath.path
		self = path.count == 0 ? "" : {
			var str = path.first!
			for i in 1..<path.count {
				str += " --> \(path[i])"
			}
			return str
		}()
	}
}

