//******************************************************************************
//  Created by Edward Connell on 08/08/17
//  Copyright © 2017 Connell Research. All rights reserved.
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
