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
import Dispatch

public protocol ModelSolver : ModelElementContainer {
	var elementToSolve: String { get set }
	var currentEpoch: Double { get }
	var maxEpochs: Double { get set }
	var currentIteration: Int { get }
	var maxIterations: Int { get set }
	var tests: OrderedNamespace<Test> { get }

	func solve() throws
	func update(parameter: LearnedParameter, using stream: DeviceStream) throws
}

public typealias TrainingCompletionHandler = (ModelSolver, Error?) -> Void

//==============================================================================
// Solver
//   solves using stochastic gradient descent
final public class Solver : ModelElementContainerBase, ModelSolver, InitHelper {
	//----------------------------------------------------------------------------
	// enums
	public enum LearningRatePolicy : String, EnumerableType {
		case exp, fixed, inverse, sigmoid, step, poly
	}
	public enum Regularization : String, EnumerableType {	case L1, L2 }

	//----------------------------------------------------------------------------
	// properties
	public var batchSize = 64                     { didSet{onSet("batchSize")} }
	public var currentEpoch = 0.0                 { didSet{onSet("currentEpoch")} }
	public var currentIteration = 0               { didSet{onSet("currentIteration")} }
	public var currentLearningRate = 0.0          { didSet{onSet("currentLearningRate")} }
	public var currentAverageLoss = 0.0           { didSet{onSet("currentAverageLoss")} }
	public var displayInterval = 100              { didSet{onSet("displayInterval")} }
	public var elementToSolve = ""                { didSet{onSet("elementToSolve")} }
	public var gamma = 0.0001                     { didSet{onSet("gamma")} }
	public var learningRate = 0.01                { didSet{onSet("learningRate")} }
	public var learningRatePolicy = LearningRatePolicy.inverse
		                                            { didSet{onSet("learningRatePolicy")} }
	public var maxEpochs: Double = 100            { didSet{onSet("maxEpochs")} }
	public var maxIterations = 10000              { didSet{onSet("maxIterations")} }
	public var momentum: Double? = 0.9            { didSet{onSet("momentum")} }
	public var objectiveStopDelta = 1e-8          { didSet{onSet("objectiveStopDelta")} }
	public var power = 0.75                       { didSet{onSet("power")} }
	public var regularization = Regularization.L2 { didSet{onSet("regularization")} }
	public var removeDatabases = false            { didSet{onSet("removeDatabases")} }
	public var stepSize = 1                       { didSet{onSet("stepSize")} }
	public var snapShotInterval: Int?             { didSet{onSet("snapShotInterval")} }
	public var testBatchSize = 100                { didSet{onSet("testBatchSize")} }
	public var testInterval = 1000                { didSet{onSet("testInterval")} }
	public var tests = OrderedNamespace<Test>()   { didSet{onSet("tests")} }
	public var weightDecay = 0.006                { didSet{onSet("weightDecay")} }

	// local data
	private var topTest: Test?
	private var topTestScore = 0.0
	private var currentLossDelta = Double.greatestFiniteMagnitude
	private var batchLosses = [Double]()
	private let batchLossHistoryCount = 100

	//----------------------------------------------------------------------------
	// addAccessors
	public override func addAccessors() {
		super.addAccessors()
		addAccessor(name: "batchSize",
		            get: { [unowned self] in self.batchSize },
		            set: { [unowned self] in self.batchSize = $0 })
		addAccessor(name: "currentEpoch",
		            get: { [unowned self] in self.currentEpoch },
		            set: { [unowned self] in self.currentEpoch = $0 })
		addAccessor(name: "currentIteration",
		            get: { [unowned self] in self.currentIteration },
		            set: { [unowned self] in self.currentIteration = $0 })
		addAccessor(name: "currentLearningRate",
		            get: { [unowned self] in self.currentLearningRate },
		            set: { [unowned self] in self.currentLearningRate = $0 })
		addAccessor(name: "currentAverageLoss",
		            get: { [unowned self] in self.currentAverageLoss },
		            set: { [unowned self] in self.currentAverageLoss = $0 })
		addAccessor(name: "elementToSolve",
		            get: { [unowned self] in self.elementToSolve },
		            set: { [unowned self] in self.elementToSolve = $0 })
		addAccessor(name: "gamma",
		            get: { [unowned self] in self.gamma },
		            set: { [unowned self] in self.gamma = $0 })
		addAccessor(name: "learningRate",
		            get: { [unowned self] in self.learningRate },
		            set: { [unowned self] in self.learningRate = $0 })
		addAccessor(name: "learningRatePolicy",
		            get: { [unowned self] in self.learningRatePolicy },
		            set: { [unowned self] in self.learningRatePolicy = $0 })
		addAccessor(name: "maxEpochs",
		            get: { [unowned self] in self.maxEpochs },
		            set: { [unowned self] in self.maxEpochs = $0 })
		addAccessor(name: "maxIterations",
		            get: { [unowned self] in self.maxIterations },
		            set: { [unowned self] in self.maxIterations = $0 })
		addAccessor(name: "momentum",
		            get: { [unowned self] in self.momentum },
		            set: { [unowned self] in self.momentum = $0 })
		addAccessor(name: "objectiveStopDelta",
		            get: { [unowned self] in self.objectiveStopDelta },
		            set: { [unowned self] in self.objectiveStopDelta = $0 })
		addAccessor(name: "power",
		            get: { [unowned self] in self.power },
		            set: { [unowned self] in self.power = $0 })
		addAccessor(name: "regularization",
		            get: { [unowned self] in self.regularization },
		            set: { [unowned self] in self.regularization = $0 })
		addAccessor(name: "removeDatabases",
		            get: { [unowned self] in self.removeDatabases },
		            set: { [unowned self] in self.removeDatabases = $0 })
		addAccessor(name: "stepSize",
		            get: { [unowned self] in self.stepSize },
		            set: { [unowned self] in self.stepSize = $0 })
		addAccessor(name: "snapShotInterval",
		            get: { [unowned self] in self.snapShotInterval },
		            set: { [unowned self] in self.snapShotInterval = $0 })
		addAccessor(name: "testBatchSize",
		            get: { [unowned self] in self.testBatchSize },
		            set: { [unowned self] in self.testBatchSize = $0 })
		addAccessor(name: "testInterval",
		            get: { [unowned self] in self.testInterval },
		            set: { [unowned self] in self.testInterval = $0 })
		addAccessor(name: "weightDecay",
		            get: { [unowned self] in self.weightDecay },
		            set: { [unowned self] in self.weightDecay = $0 })
		addAccessor(name: "tests",
		            get: { [unowned self] in self.tests },
		            set: { [unowned self] in self.tests = $0 })
	}

	//----------------------------------------------------------------------------
	// onDefaultsChanged
	public override func onDefaultsChanged() {
		super.onDefaultsChanged()
		properties["tests"]!.updateDefaults()
	}

	//----------------------------------------------------------------------------
	// setup
	// the final container should make the connections
	public override func setup(taskGroup: TaskGroup) throws {
		// find all databases and remove them if requested
		if removeDatabases {
			enumerateObjects {
				if let provider = $0 as? DbProvider {
					try? provider.removeDatabase()
				}
				return true
			}
		}
		
		// setup dependents building any necessary databases
		try super.setup(taskGroup: taskGroup)
		
		// connect
		try connectElements()
	}

	//----------------------------------------------------------------------------
	// updateAverageLoss
	private func updateAverageLoss(loss: Double) {
		// add to list
		if batchLosses.count >= batchLossHistoryCount { batchLosses.removeFirst() }
		batchLosses.append(loss)

		// reduce
		// the batchLossHistoryCount is small so we don't need to optimize this
		let newAverage = batchLosses.reduce(0, +) / Double(batchLosses.count)

		currentLossDelta = abs(newAverage - currentAverageLoss)
		currentAverageLoss = newAverage
	}

	//----------------------------------------------------------------------------
	// updateLearnedParameters
	//  this updates the parameter gradients
	private func updateLearnedParameters() throws {
		for parameter in model.learnedParameters.values {
			try parameter.updateGradient(parameter.updateStream)
			try update(parameter: parameter, using: parameter.updateStream)
		}

		// These are independent calculations, but DispatchQueue.concurrentPerform
		// is slower and must have a lot of overhead

//		let parameters = Array(model.learnedParameters.values)
//		DispatchQueue.concurrentPerform(iterations: model.learnedParameters.values.count) {
//			let parameter = parameters[$0]
//			try? parameter.updateGradient(parameter.updateStream)
//			try? update(parameter: parameter, using: parameter.updateStream)
//		}
	}

	//----------------------------------------------------------------------------
	// solve
	public func solve() throws {
		// find the training item
		var trainingItem = elementToSolve
		if let data = self.data, trainingItem.isEmpty {
			trainingItem = data.components(separatedBy: ".")[0]
		}
		guard let element = items[trainingItem] as? ModelElementContainer else {
			writeLog("\(namespaceName): elementToSolve '\(trainingItem)' not found")
			throw ModelError.evaluationFailed
		}

		// Begin
		if willLog(level: .status) {
			writeLog("", level: .status, trailing: "*")
			writeLog("Begin training: \(trainingItem)", level: .status)
		}

		// reset metrics
		_ = try resetMetricsElements(function: self)
		topTestScore = Double.greatestFiniteMagnitude

		// create selection
		var selection = Selection(count: batchSize, wrapAround: true)

		// determine when to stop
		let testGroup = DispatchGroup()
		var testPass  = 0
		currentAverageLoss = 0
		currentIteration = 0
		currentEpoch = 0

		let start = Date()
		while currentIteration < maxIterations && currentEpoch < maxEpochs {
			// forward
			let status = try element.forward(mode: .training, selection: selection)
			currentEpoch = status.epoch

			// update gradients for learned parameters
			try element.backward(solver: self, selection: selection)

			// update the parameters
			try updateLearningRate(iteration: currentIteration)
			try updateLearnedParameters()


			// test after at least one forward pass so that test templates
			// can benefit from possible cached data during setupForward
			if currentIteration > 0 && testInterval > 0 &&
						 (currentIteration % testInterval) == 0 &&
						 currentIteration < maxIterations - 1 {

				try runTests(group: testGroup, loss: currentAverageLoss,
					           epoch: currentEpoch, pass: testPass)
				testPass += 1
			}

			// advance to next
			currentIteration += 1
			selection = selection.next()
		}

		// final test
		try runTests(group: testGroup, loss: currentAverageLoss,
			           epoch: currentEpoch, pass: testPass)
		testGroup.wait()

		// run is complete
		let end = Date()
		if willLog(level: .status) {
			writeLog("Training complete. Elapsed time: " +
				String(timeInterval: end.timeIntervalSince(start)), level: .status)
			writeLog("", level: .status)
			writeLog("", level: .status, trailing: "*")
		}

		// report
		if willLog(level: .status) {
			writeLog("Top test results", level: .status)

			for test in tests {
				for item in test.topScoreItems {
					let top = test.topScores[item.key]!
					let errorStr = String(format: "%\(7).\(5)f", (1.0 - top.testError) * 100)
					let epochStr = String(format: "%\(6).\(3)f", top.epoch)
					self.writeLog("\(item.key) accuracy: \(errorStr)% epoch: \(epochStr)",
						level: .status, indent: 1)
				}
			}
			writeLog("", level: .status)
			writeLog("", level: .status, trailing: "*")
		}

		// make sure any async tasks are complete, such as
		// the database prefetch thread in streamOutput mode
		model.waitForAsyncTasksToComplete()
	}

	//----------------------------------------------------------------------------
	// update
	public func update(parameter: LearnedParameter, using stream: DeviceStream) throws {
		// apply regularization
		try regularize(parameter: parameter, using: stream)

		// update learning rate
		let rate = currentLearningRate * parameter.learningRateScale

		// get flat reference views
		var data = try parameter.data.referenceFlattened(using: stream)

		if let momentum = self.momentum {
			// lazy init
			if parameter.history.count == 0 {
				var hist = DataView(shape: parameter.data.shape, dataType: parameter.data.dataType)
				hist.name = "\(parameter.namePath).history"
				parameter.history.append(hist)
			}

			// step = hist[i] = learningRate * grad[i] + momentum * hist[i]
			// weights[i] -= step
			var hist = try parameter.history[0].referenceFlattened(using: stream)
			try stream.update(weights: &data, gradient: parameter.grad.flattened(),
				                learningRate: rate, history: &hist, momentum: momentum)
		} else {
			// weights = weights + grad * learningRate
			try stream.update(weights: &data, gradient: parameter.grad.flattened(),
				                learningRate: rate)
		}
	}

	//----------------------------------------------------------------------------
	// updateLearningRate
	func updateLearningRate(iteration: Int) throws {

		switch learningRatePolicy {
		case .exp:
			currentLearningRate = learningRate * pow(gamma, Double(iteration))
			
		case .fixed:
			currentLearningRate = learningRate
			
		case .inverse:
			currentLearningRate = learningRate * pow(1 + gamma * Double(iteration), -power)

		case .poly:
			currentLearningRate =
				learningRate * pow(1 - (Double(iteration) / Double(maxIterations)), power)
			
		case .sigmoid:
			currentLearningRate =
				learningRate * (1 / (1 + exp(-gamma * Double(iteration - stepSize))))
			
		case .step:
			currentLearningRate = learningRate * pow(gamma, Double(iteration / stepSize))
		}
	}
	
	//----------------------------------------------------------------------------
	// regularize
	private func regularize(parameter: LearnedParameter,
	                        using stream: DeviceStream) throws {
		// flatten to a vector
		let data = parameter.data.flattened()
		var grad = try parameter.grad.referenceFlattened(using: stream)

		switch regularization {
		case .L1:
			// TODO
			fatalError("not implemented")
			
		case .L2:
			try stream.axpy(alpha: weightDecay, x: data, y: &grad)
		}
	}

	//----------------------------------------------------------------------------
	// runTests
	//
	func runTests(group: DispatchGroup, loss: Double, epoch: Double, pass: Int) throws {
		// Note: maybe create solver task group for tests
		for test in tests {
			for testFunction in test.items {

				// re-setup so that template application will copy source
				// learned parameters and properties
				try testFunction.setup(taskGroup: model.tasks)

				//	The first time through we run the originals sequentially
				// to initialize data, so the copies can avoid recompute
				if model.concurrencyMode == .concurrent && pass > 0 {
					// copy and setup
					let queue = DispatchQueue(label: "testScoresQueue")
					let function = testFunction.copy(type: ModelElementContainer.self)
					function.setContext(parent: testFunction.parent!,
						typePath: "", propPath: "", namePath: function.namespaceName)

					try function.setup(taskGroup: model.tasks)
					function.name = "\(function.namespaceName):\(pass)"

					// run async
					DispatchQueue.global().async(group: group) { [unowned self] in
						let testError = self.runTest(
							function: function, loss: loss, epoch: epoch, pass: pass)

						// keep top
						queue.sync {
							let top = test.topScores[testFunction.namespaceName]
							if top == nil || testError < top!.testError {
								test.topScores[testFunction.namespaceName] = TestScore(epoch: epoch, testError: testError)
								test.topScoreItems[testFunction.namespaceName] = function
							}
						}
					}
				} else {
					let testError = runTest(
						function: testFunction as! ModelElementContainer,
						loss: loss, epoch: epoch, pass: pass)

					// keep top
					let top = test.topScores[testFunction.namespaceName]
					if top == nil || testError < top!.testError {
						test.topScores[testFunction.namespaceName] = TestScore(epoch: epoch, testError: testError)
						test.topScoreItems[testFunction.namespaceName] = testFunction
					}
				}
			}
		}
	}

	//----------------------------------------------------------------------------
	// resetFunctionMetrics
	private func resetMetricsElements(function: ModelElementContainer) throws -> [MetricsElement] {
		let elts = try function.filter { $0 is MetricsElement } as! [MetricsElement]
		for elt in elts { elt.resetMetrics() }
		return elts
	}
	
	//----------------------------------------------------------------------------
	// runTest
	//  this runs an individual test
	private func runTest(function: ModelElementContainer,
	                     loss: Double, epoch: Double, pass: Int) -> Double {
		var testError = 0.0

		do {
			let metricElements = try resetMetricsElements(function: function)

			// iterate once through the data set (one currentEpoch)
			var status = FwdProgress()
			while status.epoch < 1.0 {
				let selection = Selection(count: self.testBatchSize)
				status = try function.forward(mode: .inference, selection: selection)
			}

			// report accuracy
			for elt in metricElements {
				if let accElt = elt as? AccuracyElement {
					testError = accElt.error
					if willLog(level: .status) {
						let errorStr = String(format: "%\(8).\(2)f", accElt.accuracy * 100)
						let epochStr = String(format: "%\(6).\(3)f", epoch)

						if loss == 0 {
							self.writeLog("Test accuracy: \(errorStr)%  epoch: \(epochStr)  func: " +
								"\(function.namespaceName)", level: .status)

						} else {
							let lossStr  = String(format: "%\(8).\(6)f", loss)
							self.writeLog("Test accuracy: \(errorStr)%  " +
								"loss: \(lossStr)  epoch: \(epochStr)  func: " +
								"\(function.namespaceName)", level: .status)
						}
					}
				}
			}
		} catch {
			self.writeLog("Test: '\(function.namespaceName)' failed. " +
					"Error: \(String(describing: error))")
		}
		return testError
	}
} // Solver






