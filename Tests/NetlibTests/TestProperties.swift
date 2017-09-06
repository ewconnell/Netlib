//******************************************************************************
//  Created by Edward Connell on 04/29/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
import Foundation
import XCTest

@testable import Netlib

class TestProperties: XCTestCase
{
	static var allTests : [(String, (TestProperties) -> () throws -> Void)] {
		return [
			("test_copyModel"               , test_copyModel),
			("test_copyModelPerformance"    , test_copyModelPerformance),
			("test_getSetProperty"          , test_getSetProperty),
			("test_getPropertyPerformance"  , test_getPropertyPerformance),
			("test_setPropertyPerformance"  , test_setPropertyPerformance),
			("test_lookupDefault"           , test_lookupDefault),
			("test_lookupDefaultPerformance", test_lookupDefaultPerformance),
			("test_update"                  , test_update),
		]
	}
	
	let testString = "Test string"
	
	//----------------------------------------------------------------------------
	// test_copyModel
	func test_copyModel() {
		guard let path = getBundle(for: type(of: self)).path(
			forResource: "samples/unitTestModels/loadTest", ofType: "xml")
			else { XCTFail(); return }
		
		do {
			// load a model
			let model = try Model(contentsOf: URL(fileURLWithPath: path))
			let jsonModel = try model.asJson(after: -1)
			let other = model.copy()
			let jsonOther = try other.asJson(after: -1)
			checkDiff(jsonModel, jsonOther)
			XCTAssertTrue(jsonModel == jsonOther)
			
			let other2 = model.copy()
			let jsonOther2 = try other2.asJson(after: -1)
			checkDiff(jsonModel, jsonOther2)
			XCTAssertTrue(jsonModel == jsonOther2)
			
		} catch {
			XCTFail(String(describing: error))
		}
	}
	
	//----------------------------------------------------------------------------
	// test_copyModelPerformance
	func test_copyModelPerformance() {
		guard let path = getBundle(for: type(of: self)).path(
			forResource: "samples/unitTestModels/loadTest", ofType: "xml")
			else { XCTFail(); return }
		
		do {
			// load a model
			let model = try Model(contentsOf: URL(fileURLWithPath: path))
			
			self.measure {
				for _ in 0..<10 {
					let _ = model.copy()
				}
			}
		} catch {
			XCTFail(String(describing: error))
		}
	}
	
	//----------------------------------------------------------------------------
	// test_getSetProperty
	func test_getSetProperty() {
		let fn = Function()
		fn.name = testString
		XCTAssertTrue(fn.name == testString)
		
		let cv = Convolution()
		cv.forwardAlgorithm = .implicitGEMM
		XCTAssertTrue(cv.forwardAlgorithm == .implicitGEMM)
		
		// test optionals
		cv.bias.uri = nil
		XCTAssertTrue(cv.bias.uri == nil)
		
		let uri = Uri()
		cv.bias.uri = uri
		XCTAssertTrue(cv.bias.uri! == uri)
	}
	
	//----------------------------------------------------------------------------
	// test_getPropertyPerformance
	func test_getPropertyPerformance() {
		let cv = Convolution()
		cv.bias.uri = Uri()
		
		// 1000 set and gets
		self.measure {
			for _ in 0..<1000 {
				let _ = cv.name
				let _ = cv.name
				let _ = cv.logLevel
				let _ = cv.forwardAlgorithm
				let _ = cv.backwardDataAlgorithm
				let _ = cv.backwardDataAlgorithm
				let _ = cv.bias.name
				let _ = cv.bias.logLevel
				let _ = cv.filterSize
				let _ = cv.pad
				let _ = cv.bias.uri?.name
				let _ = cv.bias.uri?.name
			}
		}
	}
	
	//----------------------------------------------------------------------------
	// test_setPropertyPerformance
	func test_setPropertyPerformance() {
		let cv = Convolution()
		cv.bias.uri = Uri()
		
		// 1000 set and gets
		self.measure {
			for _ in 0..<1000 {
				cv.name = self.testString
				cv.name = self.testString
				cv.logLevel = .warning
				cv.forwardAlgorithm = .implicitGEMM
				cv.backwardDataAlgorithm = .algo0
				cv.backwardDataAlgorithm = .algo0
				cv.bias.name = self.testString
				cv.bias.logLevel = .diagnostic
				cv.filterSize = [5, 5]
				cv.pad = [2, 2, 2]
				cv.bias.uri?.name = self.testString
				cv.bias.uri?.name = self.testString
			}
		}
	}
	
	//----------------------------------------------------------------------------
	// test_lookupDefault
	func test_lookupDefault() {
		let model = Model()
		let pool = model.items.append(Pooling())
		
		// verify initial default value
		XCTAssert(pool.mode == .max)
		
		model.defaults = [Default { $0.property = ".mode"; $0.value = "averageExcludePadding"}]
		let newVal = pool.mode
		XCTAssert(newVal == .averageExcludePadding,
		          "found \(newVal), expected averageExcludePadding")
	}
	
	func test_lookupDefaultPerformance() {
		self.measure {
			for _ in 0..<1000 {
			}
		}
	}
	
	//----------------------------------------------------------------------------
	// test_update
	func test_update() {
		do {
			let master = Model()
			let masterFn = master.items.append(Function())
			masterFn.name = "fn"
			let masterCv = masterFn.items.append(Convolution())
			masterCv.name = "cv"
			let client = Model()
			
			// set some props on the Master model
			let modelName = "Test Model"
			master.name = modelName
			masterFn.logLevel = .warning
			masterCv.forwardAlgorithm = .implicitGEMM
			masterCv.bias.uri = Uri()
			masterCv.bias.uri?.name = "Uri Name"
			
			// synchronize client and master
			try client.updateAny(with: master.selectAny(after: client.version))
			
			// set some props on the client
			let clientCv = client.find(type: Convolution.self, name: "cv")!
			let clientFn = client.find(type: Function.self, name: "fn")!
			clientCv.backwardDataAlgorithm = .algo1
			let con = Connector(name: "input", connection: Connection(input: "database"))
			clientCv.inputs["input"] = con
			
			// synchronize client and master
			try master.updateAny(with: client.selectAny(after: master.version))
			
			// compare
			XCTAssert(client.name == master.name)
			XCTAssert(clientFn.logLevel == masterFn.logLevel)
		} catch {
			XCTFail(String(describing: error))
		}
	}
}
