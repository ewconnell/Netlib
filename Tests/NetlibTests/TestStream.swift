//******************************************************************************
//  Created by Edward Connell on 1/9/2017.
//  Copyright © 2016 Connell Research. All rights reserved.
//
import XCTest
import Foundation

@testable import Netlib

class TestStream: XCTestCase
{
	static var allTests : [(String, (TestStream) -> () throws -> Void)] {
		return [
			("test_compareEqual", test_compareEqual),
		]
	}
	
	//----------------------------------------------------------------------------
	// test_compareEqual
	func test_compareEqual() {
		do {
			let model = Model()
			try model.setup()
			let stream = try model.compute.requestStreams(label: "dataStream")[0]
			
			var aData = DataView(count: 10)
			var bData = DataView(count: 10)
			var result = DataView(count: 10)
			
			try stream.fillWithIndex(data: &aData, startingAt: 0)
			try cpuFillWithIndex(data: &bData, startingAt: 0)
			try bData.set(value: 0, at: [2])
			try bData.set(value: 0, at: [5])
			try bData.set(value: 0, at: [9])
			
			try stream.compareEqual(data: aData, with: bData, result: &result)
			
			var sum = DataView(count: 1)
			try stream.asum(x: result, result: &sum)
			XCTAssert(try sum.get() as Int == 7)
			
		} catch {
			print(String(describing: error))
		}
	}
}
