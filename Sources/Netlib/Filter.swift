//******************************************************************************
//  Created by Edward Connell on 7/17/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
import Foundation
import Dispatch

//==============================================================================
// FilterBase
public protocol Filter : ModelElement {
	var dataType: DataType? { get set }
	var outDataType: DataType { get }
}

open class FilterBase : ModelElementBase, Filter {
	// initializers
	public required init() {
		super.init()
		var prop = properties["inputs"]!
		prop.isGenerated = true
		prop.isCopyable = false
		prop = properties["outputs"]!
		prop.isGenerated = true
		prop.isCopyable = false
	}

	public func onSetupData(using stream: DeviceStream) throws {}

	//----------------------------------------------------------------------------
	// properties
	public var dataType: DataType?                 { didSet{onSet("dataType")} }
	public var outDataType: DataType = .real32F
	public var inputConnector: Connector!
	public var labelsInputConnector: Connector?
	public private(set) var dataConnector = Connector(name: "data")
	private var lossSum: DataView!

	//----------------------------------------------------------------------------
	// addAccessors
	public override func addAccessors() {
		super.addAccessors()
		addAccessor(name: "dataType",
			get: { [unowned self] in self.dataType },
			set: { [unowned self] in self.dataType = $0 })
	}

	//----------------------------------------------------------------------------
	// setup
	public override func setup(taskGroup: TaskGroup) throws {
		// templates are applied during this phase
		try super.setup(taskGroup: taskGroup)

		// add filter 'data' connector
		outputs["data"] = dataConnector

		// add connectors if specified through attributes
		if let inputName = input {
			inputConnector = Connector(name: "input", connection: Connection(input: inputName))
		} else {
			inputConnector = Connector(name: "input", minConnections: 1, maxConnections: 1)
		}
		inputs["input"] = inputConnector

		if let labelsInputName = labelsInput {
			labelsInputConnector = Connector(name: "labelsInput",
				connection: Connection(input: labelsInputName),	minConnections: 0)
			inputs["labelsInput"] = labelsInputConnector!
		}
	}

	//----------------------------------------------------------------------------
	// setupForward
	//	create/setup a computable for each device stream
	public override func setupForward(mode: EvaluationMode, selection: Selection) throws {
		try super.setupForward(mode: mode, selection: selection)

		// determine output data type
		outDataType = try dataType ?? getInput().items[0].data.dataType

		try onSetupData(using: getInput().items[0].stream)

		// reinitialize in case of multiple setup
		dataConnector.items.removeAll()
	}

	//----------------------------------------------------------------------------
	// getInput
	public func getInput() throws -> Connector {
		// validate
		guard inputConnector.dependents.count == 1 else {
			writeLog("\(namespaceName).input requires 1 connection." +
				" \(self.inputConnector.dependents.count) found")
			writeLog("path to '\(namespaceName)' = \(self.namePath)")
			throw ModelError.evaluationFailed
		}

		guard inputConnector.dependents[0].items.count != 0 else {
			let elementName = inputConnector.dependents[0].element.namespaceName
			let connectorName = inputConnector.dependents[0].namespaceName
			writeLog("\(namespaceName).input: \(elementName).\(connectorName)" +
				" '.data' is nil")
			throw ModelError.evaluationFailed
		}
		return inputConnector.dependents[0]
	}

	//----------------------------------------------------------------------------
	// getLabelsInput
	public func getLabelsInput() -> Connector? {
		return labelsInputConnector?.dependents.isEmpty == true ? nil :
			labelsInputConnector?.dependents[0]
	}


	//----------------------------------------------------------------------------
	// onComputeLoss
	public override func onComputeLoss() throws -> Double {
		// use the input grad and data
		let item = try getInput().items[0]
		guard let grad = item.grad else { return 0 }

		// lazy allocate
		if lossSum == nil {
			lossSum = DataView(count: 1, dataType: item.data.dataType)
		}
		try item.stream.asum(x: grad.flattened(), result: &lossSum!)
		return try lossSum.get() * lossWeight
	}
}

//==============================================================================
// ComputableFilterBase
public protocol ComputableFilter : Filter {}

open class ComputableFilterBase : FilterBase, ComputableFilter {
	// properties
	private var computables = [Computable]()
	private var dataError: DataView!
	private var labelError: DataView!

	//----------------------------------------------------------------------------
	// setupForward
	//	create/setup a computable for each device stream
	public override func setupForward(mode: EvaluationMode, selection: Selection) throws {
		try super.setupForward(mode: mode, selection: selection)

		// reinitialize in case of multiple setup
		computables.removeAll()

		// item references
		let inputConnector = try getInput()
		let labelsInputConnector = getLabelsInput()
		assert(inputConnector.items.count == 1)

		for i in 0..<inputConnector.items.count {
			let inItem = inputConnector.items[i]
			let labelItem = labelsInputConnector?.items[i]

			// create data output item
			let outItem = ConnectorItem(using: inItem.stream)
			dataConnector.items.append(outItem)

			// create computable and setup
			assert(self is ComputableFilterProperties)
			let computable = try inItem.stream
				.createComputable(type: typeName, props: self as! ComputableFilterProperties)
			computables.append(computable)

			// setup
			try computable.setupForward(
				mode: mode,
				inData: inItem.data,
				labels: labelItem?.data,
				outData: &outItem.data,
				backData: &outItem.backData)

			// make sure the computable isn't holding extra references
			assert(outItem.data.isShared || outItem.data.isUniqueReference(),
				"Computable is holding extra reference to outData which will cause mutation")
			assert(outItem.backData == nil || outItem.backData!.isUniqueReference())

			// make the output data shared to allow down stream
			// consumers to do a zero copy pass forward in noop situations
			outItem.data.isShared = true
		}
	}

	//----------------------------------------------------------------------------
	// onForward
	public override func onForward(mode: EvaluationMode, selection: Selection) throws {
		let inputConnector = try getInput()
		let labelsInputConnector = getLabelsInput()
		assert(!computables.isEmpty && inputConnector.items.count == computables.count)

		// TODO: concurrentPerform is slower than looping with 2 devices,
		// TODO: test if there is any benefit when more devices are involved
		for i in 0..<computables.count {
			let inItem    = inputConnector.items[i]
			let outItem   = dataConnector.items[i]
			let labelItem = labelsInputConnector?.items[i]

			// protect against accidental retained references
			assert(inItem.data.isShared, "\(inputConnector.element.typeName)." +
				"data should be sharable. isShared should be initialized to true")
			assert(labelItem == nil || labelItem!.data.isShared,
				"\(labelsInputConnector!.element.typeName)." +
					"labels should be sharable. isShared should be initialized to true")
			assert(outItem.backData == nil || outItem.backData!.isUniqueReference())

			// select the item stream device
			try inItem.stream.device.select()

			try computables[i].forward(
				mode: mode,
				inData: inItem.data,
				labels: labelItem?.data,
				outData: &outItem.data,
				backData: &outItem.backData)

			// protect against accidental retained references
			assert(outItem.backData == nil || outItem.backData!.isUniqueReference())
		}
	}

	//----------------------------------------------------------------------------
	// setupBackward
	public override func setupBackward() throws {
		let inputConnector = try getInput()
		let hasLabels = getLabelsInput() != nil

		for i in 0..<computables.count {
			let inItem = inputConnector.items[i]
			let outItem = dataConnector.items[i]

			// select the item stream device
			try outItem.stream.device.select()

			// if there are no labels, then this element performs a grad calculation
			if !hasLabels {
				outItem.grad = DataView(shape: outItem.data.shape,
				                        dataType: outItem.data.dataType)
				outItem.grad!.isShared = true
			}

			// setup computable
			try computables[i].setupBackward(
				outData: outItem.data,
				outGrad: outItem.grad,
				inData: inItem.data)
		}

		// setup dependents
		try super.setupBackward()
	}

	//----------------------------------------------------------------------------
	// onBackward
	public override func onBackward(solver: ModelSolver) throws {
		try super.onBackward(solver: solver)
		let inputConnector = try getInput()

		// get optional labels
		let labelsInputConnector = getLabelsInput()

		for i in 0..<computables.count {
			let inItem = inputConnector.items[i]
			let outItem = dataConnector.items[i]
			let labelItem = labelsInputConnector?.items[i]
			assert(inItem.grad == nil || inItem.grad!.isShared || inItem.grad!.isUniqueReference())

			// mutually exclusive
			assert((labelItem == nil) != (outItem.grad == nil))

			// select the item stream device
			try outItem.stream.device.select()

			// back propagate
			try computables[i].backward(
				outData: outItem.backData ?? outItem.data,
				outGrad: outItem.grad,
				inData: inItem.data,
				inGrad: &inItem.grad,
				solver: solver,
				labels: labelItem?.data)
		}
	}
}
