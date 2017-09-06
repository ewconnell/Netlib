//******************************************************************************
//  Created by Edward Connell on 05/24/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
import XCTest
import Foundation

@testable import Netlib

class TestLog: XCTestCase {
	static var allTests : [(String, (TestLog) -> () throws -> Void)] {
		return [
			("test_writeLog"           , test_writeLog),
			("testPerformance_writeLog", testPerformance_writeLog),
		]
	}
	
	//----------------------------------------------------------------------------
	// test_writeLog
	func test_writeLog() {
		let model = Model()
		model.logLevel = .diagnostic
		model.log.silent = true
		model.name = "mnist"
		let solver = model.items.append(Solver())
		let test   = solver.tests.append(Test { $0.name = "TrainingTest" })
		let fn     = solver.items.append(Function())
		let pool   = fn.items.append(Pooling())
		let cv     = fn.items.append(Convolution())
		cv.bias = LearnedParameter()
		cv.bias.uri = Uri()
		cv.bias.uri?.name = "http://bias.dat"
		cv.bias.uriDataExtent = [1, 1, 800, 500]
		cv.pad = [2, 2]
		
		model.logLevel = .diagnostic
		model.writeLog("Model Warning", level: .warning)
		solver.writeLog("ModelSolver diagnostic", level: .diagnostic)
		test.writeLog("Test diagnostic", level: .diagnostic)
		fn.writeLog("Function Status", level: .status)
		cv.writeLog("Convolution Error")
		pool.diagnostic("Pooling Diagnostic", categories: .setup)
	}
	
	//----------------------------------------------------------------------------
	// testPerformance_writeLog
	func testPerformance_writeLog() {
		let model = Model()
		model.logLevel = .diagnostic
		model.log.silent = true
		model.log.maxHistory = 1
		model.name = "mnist"
		
		let solver = model.items.append(Solver())
		let fn     = solver.items.append(Function())
		let pool   = fn.items.append(Pooling())
		let cv     = fn.items.append(Convolution())
		cv.bias = LearnedParameter()
		cv.bias.uri = Uri()
		cv.bias.uri?.name = "http://bias.dat"
		cv.bias.uriDataExtent = [1, 1, 800, 500]
		cv.pad = [2, 2]
		
		self.measure {
			for _ in 0..<1000 {
				model.writeLog("Model Warning", level: .warning)
				solver.writeLog("ModelSolver diagnostic", level: .diagnostic)
				solver.writeLog("ModelSolver diagnostic", level: .diagnostic)
				fn.writeLog("Function Status", level: .status)
				fn.writeLog("Function Status", level: .status)
				cv.writeLog("Convolution Error")
				cv.writeLog("Convolution Error")
				cv.writeLog("Convolution Error")
				pool.diagnostic("Pooling Diagnostic", categories: .setup)
				pool.diagnostic("Pooling Diagnostic", categories: .setup)
			}
		}
	}
}
