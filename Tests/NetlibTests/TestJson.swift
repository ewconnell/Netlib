//******************************************************************************
//  Created by Edward Connell on 05/01/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
import XCTest
import Foundation

@testable import Netlib

class TestJson: XCTestCase {
	static var allTests : [(String, (TestJson) -> () throws -> Void)] {
		return [
			("test_incrementalAddDefaults"       , test_incrementalAddDefaults),
			("test_addRemoveSync"                , test_addRemoveSync),
			("test_addRemoveSyncPrettyPrinted"   , test_addRemoveSyncPrettyPrinted),
			("test_selectChangedObjectProperties", test_selectChangedObjectProperties),
			("test_selectChangedProperties"      , test_selectChangedProperties),
			("test_selectOptionalProperties"     , test_selectOptionalProperties),
			("test_performanceSelectAny"         , test_performanceSelectAny),
		]
	}
	
	//----------------------------------------------------------------------------
	// test_incrementalAddDefaults
	func test_incrementalAddDefaults() {
		do {
			let model = Model()
			model.defaults = [Default { $0.property = ".dataType"; $0.value = "half" }]
			
			let fc = model.items.append(FullyConnected())
			fc.bias.uri = Uri(string: "./blah/data")
			
			let newModel = Model()
			try newModel.update(fromJson: model.asJson(after: newModel.version))
			
			try checkDiff(model.asJson(), newModel.asJson())
			
			// add something to model, sync, compare changes
			let version = newModel.version
			model.defaults!.append(Default { $0.property = ".unzip"; $0.value = "true" })
			try newModel.update(fromJson: model.asJson(after: version))
			try checkDiff(model.asJson(after: version), newModel.asJson(after: version))
			
			// compare whole models
			try checkDiff(model.asJson(after: -1), newModel.asJson(after: -1))
		} catch {
			XCTFail(String(describing: error))
		}
	}
	
	//----------------------------------------------------------------------------
	// test_addRemoveSync
	func test_addRemoveSync() {
		do {
			// define a master
			let master = Model()
			master.name = "mnist"
			let fn = master.items.append(Function())
			fn.items.append(Pooling())
			let cv = fn.items.append(Convolution())
			cv.bias = LearnedParameter()
			cv.bias.uri = Uri(string: "http://bias.dat")
			cv.bias.uriDataExtent = [1, 1, 800, 500]
			
			let client = Model()
			
			// sync added items
			var jsonMaster = try master.asJson(after: client.version)
			try client.update(fromJson: jsonMaster)
			var jsonClient = try client.asJson()
			checkDiff(jsonMaster, jsonClient)
			
			// remove some things
			cv.bias.uri = nil
			fn.items.remove(name: "pool")
			
			jsonMaster = try master.asJson()
			try client.update(fromJson: jsonMaster)
			jsonClient = try client.asJson()
			checkDiff(jsonClient, jsonMaster)
			
			// check all
			try checkDiff(master.asJson(after: -1), client.asJson(after: -1))
		} catch {
			XCTFail(String(describing: error))
		}
	}
	
	//----------------------------------------------------------------------------
	// test_addRemoveSyncPrettyPrinted
	func test_addRemoveSyncPrettyPrinted() {
		do {
			// define a master
			let master = Model()
			master.name = "mnist"
			let fn = master.items.append(Function())
			fn.items.append(Pooling())
			let cv = fn.items.append(Convolution())
			cv.bias = LearnedParameter()
			cv.bias.uri = Uri(string: "http://bias.dat")
			cv.bias.uriDataExtent = [1, 1, 800, 500]
			
			let client = Model()
			
			// sync added items
			var jsonMaster = try master.asJson(after: client.version, options: .prettyPrinted)
			try client.update(fromJson: jsonMaster)
			var jsonClient = try client.asJson(options: .prettyPrinted)
			checkDiff(jsonClient, jsonMaster)
			
			// remove some things
			cv.bias.uri = nil
			fn.items.remove(name: "pool")
			
			jsonMaster = try master.asJson(options: .prettyPrinted)
			try client.update(fromJson: jsonMaster)
			jsonClient = try client.asJson(options: .prettyPrinted)
			checkDiff(jsonClient, jsonMaster)
			
			// check all
			try checkDiff(client.asJson(after: -1, options: .prettyPrinted),
			              master.asJson(after: -1, options: .prettyPrinted))
		} catch {
			XCTFail(String(describing: error))
		}
	}
	
	//----------------------------------------------------------------------------
	// test_selectChangedObjectProperties
	func test_selectChangedObjectProperties() {
		let model = Model()
		var version = model.version
		
		do {
			let cv = model.items.append(Convolution())
			_ = try model.asJson(after: version, options: .prettyPrinted)
			
			version = model.version
			model.name = "model1"
			cv.name = "conv1"
			_ = try model.asJson(after: version, options: .prettyPrinted)
			
			cv.bias = LearnedParameter()
			cv.bias.uri = Uri()
			
			_ = try model.asJson(after: version, options: .prettyPrinted)
			cv.bias.uri?.name = "bias name"
			
			_ = try model.asJson(after: version, options: .prettyPrinted)
		} catch {
			XCTFail(String(describing: error))
		}
	}
	
	//----------------------------------------------------------------------------
	// test_selectChangedProperties
	func test_selectChangedProperties() {
		do {
			let model = Model()
			let version = model.version
			let cv = model.items.append(Convolution())
			cv.bias.uri = Uri()
			cv.bias.uri?.name = "http://conv1_bias.bin"
			
			_ = try model.asJson(options: .prettyPrinted)
			_ = try model.asJson(after: version, options: .prettyPrinted)
		} catch {
			XCTFail(String(describing: error))
		}
	}
	
	func test_selectOptionalProperties() {
		let learned = LearnedParameter()
		_ = learned.selectAny()
		learned.uri = Uri()
		_ = learned.selectAny()
	}
	
	//----------------------------------------------------------------------------
	// test_performanceSelectAny
	func test_performanceSelectAny() {
		guard let path = getBundle(for: type(of: self)).path(
			forResource: "samples/unitTestModels/loadTest", ofType: "xml") else
		{ XCTFail(); return }
		
		do {
			let data = try String(data: Data(contentsOf: URL(fileURLWithPath: path)),
			                      encoding: .utf8)!
			let model = Model()
			try model.update(fromXml: data)
			
			self.measure {
				for _ in 0..<100 {
					do {
						let _ = try model.asJson()
					} catch {
						XCTFail(String(describing: error))
					}
				}
			}
		} catch {
			XCTFail(String(describing: error))
		}
	}
}
