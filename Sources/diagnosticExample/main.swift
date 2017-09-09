//******************************************************************************
//  Created by Edward Connell on 08/08/17
//  Copyright © 2017 Connell Research. All rights reserved.
//
//  This creates and trains an MNIST classifier while demonstrating some diagnostics
//
import Foundation
import Netlib

// specify an object allocation id to break on
//objectTracker.debuggerRegisterBreakId = 42

func meanTime(times: [TimeInterval]) -> TimeInterval {
	// drop the first iteration to eliminate load/init times for external
	// libraries such as Cuda. The same is done in the python scripts when
	// comparing with Caffe2 and TensorFlow
	let t = times[(times.count > 1 ? 1 : 0)...]
	return t.reduce(0, +) / Double(t.count)
}


do {
	let model = Model {
//	$0.concurrencyMode = .serial
//	$0.log.categories = [.dataCopy, .dataMutation]
//	$0.log.logLevel = .diagnostic
		$0.log.logLevel = .status
	}

	let fileName = "samples/mnist/mnistClassifierSolver"
	guard let path = Bundle.main.path(forResource: fileName, ofType: "xml") else {
		fatalError("Resource not found: \(fileName)")
	}
	try model.load(contentsOf: URL(fileURLWithPath: path))

	var times = [TimeInterval]()
	for _ in 0..<5 {
		let start = Date()
		try model.train()
		times.append(Date().timeIntervalSince(start))
	}

	print("\nMean training time: \(String(timeInterval: meanTime(times: times)))")
//	try print(model.asJson(options: .prettyPrinted))


} catch {
	print(String(describing: error))
}

// optionally check to see if there are any leaked objects
if objectTracker.hasActiveObjects {
	print(objectTracker.getActiveObjectInfo())
}

//let modelName = "samples/unitTestModels/mnistForward/mnistForwardSolver"
//try model.setup()
//_ = try model.forward(mode: .inference, selection: Selection(count: 5))
//var labels = DataView()
//try model.copyOutput(connectorName: "labels", to: &labels)
//print(labels.format())

//objectTracker.debuggerRegisterBreakId = 450
