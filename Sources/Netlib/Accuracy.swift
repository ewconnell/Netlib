//******************************************************************************
//  Created by Edward Connell on 5/27/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
final public class Accuracy : FilterBase, AccuracyElement, InitHelper {
	//----------------------------------------------------------------------------
	// properties
	public var accuracy = 0.0										{ didSet{onSet("accuracy")}}
	public var error = 1.0		  								{ didSet{onSet("error")}}
	public var evaluatedCount = 0								{ didSet{onSet("evaluatedCount")}}
	private var numCorrect = 0
	private var itemReductionContexts = [ReductionContext]()
	private var sum = DataView()

	//----------------------------------------------------------------------------
	// addAccessors
	public override func addAccessors() {
		super.addAccessors()
		addAccessor(name: "accuracy",
		            get: { [unowned self] in self.accuracy },
		            set: { [unowned self] in self.accuracy = $0 })
		addAccessor(name: "error",
		            get: { [unowned self] in self.error },
		            set: { [unowned self] in self.error = $0 })
		addAccessor(name: "evaluatedCount",
		            get: { [unowned self] in self.evaluatedCount },
		            set: { [unowned self] in self.evaluatedCount = $0 })
	}

	//----------------------------------------------------------------------------
	// setupForward
	public override func setupForward(mode: EvaluationMode, selection: Selection) throws {
		try super.setupForward(mode: mode, selection: selection)
		guard getLabelsInput() != nil else {
			writeLog("\(namespaceName) requires labelsInput")
			throw ModelError.setupFailed
		}

		// init member variables
		resetMetrics()

		// allocate result buffers
		let inputItems = try getInput().items
		let dataType = DataType.real32F // inputItems[0].data.dataType
		sum = DataView(count: 1, dataType: dataType)
		let sumTensorShape = Shape(extent: [1, 1, 1, 1], layout: .nchw)

		dataConnector.items.removeAll(keepingCapacity: true)
		itemReductionContexts.removeAll(keepingCapacity: true)

		for i in 0..<inputItems.count {
			let item = ConnectorItem(using: inputItems[i].stream)
			item.data = DataView(shape: inputItems[i].data.shape, dataType: dataType)
			dataConnector.items.append(item)

			// create reduction context
			let itemTensorShape = Shape(extent: [item.data.elementCount, 1, 1, 1], layout: .nchw)
			let context = try item.stream.createReductionContext(
				op: .add, dataType: dataType,
				inShape: itemTensorShape, outShape: sumTensorShape)

			itemReductionContexts.append(context)
		}
	}

	//----------------------------------------------------------------------------
	// onForward
	public override func onForward(mode: EvaluationMode, selection: Selection) throws	{
		let inputConnector = try getInput()
		let labelsInputConnector = getLabelsInput()!

		for i in 0..<inputConnector.items.count {
			let outputItem = dataConnector.items[i]
			let inputItem = inputConnector.items[i]
			let labelItem = labelsInputConnector.items[i]
			let stream = inputItem.stream

			// increment the count
			evaluatedCount += inputItem.data.extent[0]

			// compare
			try stream.compareEqual(data: inputItem.data, with: labelItem.data,
			                        result: &outputItem.data)

			// reduce
			try stream.reduce(context: itemReductionContexts[i],
				                inData: outputItem.data.flattened(), outData: &sum)
			numCorrect += try sum.get() as Int
		}

		accuracy = Double(numCorrect) / Double(evaluatedCount)
		error = 1.0 - accuracy
	}

	//----------------------------------------------------------------------------
	// resetMetrics
	public func resetMetrics() {
		evaluatedCount = 0
		numCorrect = 0
	}
}

