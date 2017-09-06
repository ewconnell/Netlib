//******************************************************************************
//  Created by Edward Connell on 05/17/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
import Foundation
import XCTest

@testable import Netlib

class TestXml: XCTestCase {
	static var allTests : [(String, (TestXml) -> () throws -> Void)] {
		return [
			("test_reloadXml"             , test_reloadXml),
			("test_loadXmlFile"           , test_loadXmlFile),
			("test_loadXmlFilePerformance", test_loadXmlFilePerformance),
		]
	}
	
	//----------------------------------------------------------------------------
	// test_reloadXml
	func test_reloadXml() {
		let model = Model()
		model.defaults = [Default]()
		model.defaults!.append(Default { $0.property = ".dataType"; $0.value = "half" })
		model.defaults!.append(Default { $0.property = ".unzip"; $0.value = "false"})
		
		do {
			let fn = model.items.append(Function())
			fn.items.append(FullyConnected())
			let cv = fn.items.append(Convolution())
			cv.bias.uri = Uri()
			
			// create a new model from the serialized model
			let newModel = Model()
			let setXml = model.asXml(after: 0, format: true, tabSize: 2)
			try newModel.update(fromXml: setXml)
			
			// compare whole models
			let originalXml = model.asXml(after: -1, format: true, tabSize: 2)
			let newXml = newModel.asXml(after: -1, format: true, tabSize: 2)
			checkDiff(originalXml, newXml)
		} catch {
			XCTFail(String(describing: error))
		}
	}
	
	//----------------------------------------------------------------------------
	// test_loadXmlFile
	func test_loadXmlFile() {
		guard let path = getBundle(for: type(of: self)).path(
			forResource: "samples/mnist/mnistClassifierSolver", ofType: "xml")
			else { XCTFail("load resource failed"); return	}
		
		do {
			let model = try Model(contentsOf: URL(fileURLWithPath: path))
			let _ = model.asXml(after: 0, format: true, tabSize: 2)
		} catch {
			XCTFail(String(describing: error))
		}
	}
	
	//----------------------------------------------------------------------------
	// test_loadXmlFilePerformance
	func test_loadXmlFilePerformance() {
		guard let path = getBundle(for: type(of: self)).path(
			forResource: "samples/unitTestModels/loadTest", ofType: "xml")
			else { XCTFail("load resource failed"); return	}
		
		self.measure {
			do {
				for _ in 0..<10 {
					let _ = try Model(contentsOf: URL(fileURLWithPath: path))
				}
			} catch {
				XCTFail(String(describing: error))
			}
		}
	}
	
} // end class
