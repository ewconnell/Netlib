//******************************************************************************
//  Created by Edward Connell on 08/08/17
//  Copyright © 2017 Connell Research. All rights reserved.
//
//  This creates and trains an MNIST classifier using an xml model definition
//
import Foundation
import Netlib

do {
	let fileName = "samples/mnist/mnistClassifierSolver"
	guard let path = Bundle.main.path(forResource: fileName, ofType: "xml") else {
		fatalError("Resource not found: \(fileName)")
	}

	let model = try Model(contentsOf: URL(fileURLWithPath: path))
	try model.train()

} catch {
	print(String(describing: error))
}
