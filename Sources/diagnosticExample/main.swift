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
//  This creates and trains an MNIST classifier while demonstrating some diagnostics
//
import Foundation
import Netlib

// specify an object allocation id to break on
//objectTracker.debuggerRegisterBreakId = 42

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
	try model.train()

	// mean training time test
	let numberOfRuns = 4
	let start = Date()
	for _ in 0..<numberOfRuns {
		try model.train()
	}
	let end = Date()
	print("Mean time: " + String(timeInterval: end.timeIntervalSince(start) / Double(numberOfRuns)))

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
