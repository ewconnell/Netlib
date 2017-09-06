//******************************************************************************
//  Created by Edward Connell on 6/7/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
import XCTest
import Foundation

@testable import Netlib

class TestSetup: XCTestCase
{
	static var allTests : [(String, (TestSetup) -> () throws -> Void)] {
		return [
			("test_mnistForward"      , test_mnistForward),
			("test_mnistForwardSolver", test_mnistForwardSolver),
		]
	}
	
	//----------------------------------------------------------------------------
	// test_mnistForward
	func test_mnistForward() {
		let modelName = "samples/unitTestModels/mnistForward/mnistForward"
		guard let path = getBundle(for: type(of: self)).path(
			forResource: modelName, ofType: "xml")
			else { XCTFail("unable to find \(modelName)"); return }
		
		do {
			let model = try Model(contentsOf: URL(fileURLWithPath: path))
			//			model.concurrencyMode = .serial
			try model.setup()
			
			// forward test
			_ = try model.forward(mode: .inference, selection: Selection(count: 10))
			
			var labels = DataView()
			try model.copyOutput(connectorName: "labels", to: &labels)
			
			var data = DataView()
			try model.copyOutput(connectorName: "data", to: &data)
			
			if data.items != labels.items {
				XCTFail("data and labels counts do not match")
				return
			}
			
			for i in 0..<data.elementCount {
				let label: Float = try labels.get(at: [i])
				let value: Float = try data.get(at: [i])
				if(!almostEquals(label, value, tolerance: 0.0001)) {
					XCTFail("values do not match")
					break
				}
			}
		} catch {
			XCTFail(String(describing: error))
		}
	}

	//----------------------------------------------------------------------------
	// test_mnistForwardSolver
	func test_mnistForwardSolver() {
		let modelName = "samples/unitTestModels/mnistForward/mnistForwardSolver"
		guard let path = getBundle(for: type(of: self)).path(
			forResource: modelName, ofType: "xml")
			else { XCTFail("unable to find \(modelName)"); return }

		do {
			let model = try Model(contentsOf: URL(fileURLWithPath: path))
			//				logLevel: .diagnostic, logCategories: [.setup, .connections])
			//			model.concurrencyMode = .serial
			try model.setup()

		} catch {
			XCTFail(String(describing: error))
		}
	}
}
