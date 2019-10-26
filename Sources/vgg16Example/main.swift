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
//  This creates and trains the VGG16 classifier
//
import Foundation
import Netlib

// specify an object allocation id to break on
//objectTracker.debuggerRegisterBreakId = 42

do {
	let model = Model {
//	$0.concurrencyMode = .serial
//	$0.log.categories = [.dataCopy, .dataMutation]
//	$0.log.categories = [.connections]
//	$0.log.logLevel = .diagnostic
		$0.log.logLevel = .status
	}

	let modelName = "samples/vggnet/vgg16Solver"
	guard let path = Bundle.main.path(forResource: modelName, ofType: "xml") else { exit(1) }
	try model.load(contentsOf: URL(fileURLWithPath: path))

	try model.train()


} catch {
	print(String(describing: error))
}

if objectTracker.hasActiveObjects {
	print(objectTracker.getActiveObjectInfo())
}
