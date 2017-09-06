//******************************************************************************
//  Created by Edward Connell on 08/08/17
//  Copyright © 2017 Connell Research. All rights reserved.
//
//  This creates and trains an MNIST classifier entirely from swift code
//
import Foundation
import Netlib

//----------------------------------------------------------------------------
// Classifier function to train/test
let mnistClassifier = Function {
	$0.name = "mnistClassifier"
	$0.labelsInput = "trainingDb.labels"

	// Data inputs are implied by collection order, or may be explicitly specified
	$0.items.append([
		Softmax { $0.labelsInput = ".labelsInput" },
		FullyConnected { $0.outputChannels = 10 },
		Activation(),
		FullyConnected { $0.outputChannels = 500 },
		Pooling(),
		Convolution { $0.outputChannels = 50; $0.filterSize = [5] },
		Pooling(),
		Convolution { $0.outputChannels = 20; $0.filterSize = [5] },
	])
}

//----------------------------------------------------------------------------
// Training model
//  - defines solver parameters
//  - references the function above to train
//  - defines training and validation databases, building them if needed
//  - defines validation tests to run
let trainingModel = Model { model in
	model.log.logLevel = .status

	model.items.append([
		Solver { solver in
			solver.testBatchSize = 1000

			// defaults override matching properties on children that are not explicitly set
			solver.defaults = [
				Default { $0.property = ".cacheDir"; $0.value = "~/Documents/data/cache/mnist/" },
				Default { $0.property = ".dataDir"; $0.value = "~/Documents/data/mnist/" },
				Default { $0.property = ".weights.fillMethod"; $0.value = "xavier" },
				Default { $0.property = "ImageCodec.format"; $0.object = ImageFormat {
					$0.encoding = .png
					$0.channelFormat = .gray }
				},
			]

			// items to train
			solver.items.append([
				mnistClassifier,
				Database {
					$0.name = "trainingDb"
					$0.connection = "trainingDb"
					$0.source = Mnist { $0.dataSet = .training }
				}
			])

			// test on validation data
			solver.tests.append([
				Test { test in
					test.items.append([
						Function {
							$0.name = "validationData"
							$0.items.append([
								Accuracy { $0.labelsInput = "validationDb.labels" },
								// this function is initialized with the current property values
								// of the template source, unless explicitly set like "labelsInput"
								Function {
									$0.template = "mnistClassifier"
									$0.labelsInput = "validationDb.labels"
								},
								Database {
									$0.name = "validationDb"
									$0.connection = "validationDb"
									$0.source = Mnist { $0.dataSet = .validation }
								},
							])
						}
					])
				}
			])
		}
	])
}

// do the training
do {
	try trainingModel.train()
} catch {
	print(String(describing: error))
}
