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
	guard let path = Bundle.main.path(forResource: fileName, ofType: "xml") else { exit(1) }

	let model = try Model(contentsOf: URL(fileURLWithPath: path), logLevel: .status)
	try model.train()

} catch {
	print(String(describing: error))
}
