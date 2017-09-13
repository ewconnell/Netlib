//******************************************************************************
//  Created by Edward Connell on 12/6/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
import XCTest
import Foundation

@testable import Netlib

class TestFill: XCTestCase {
	static var allTests : [(String, (TestFill) -> () throws -> Void)] {
		return [
			("test_fillUniform" , test_fillUniform),
			("test_fillGaussian", test_fillGaussian),
			("test_fillXavier"  , test_fillXavier),
			("test_fillMSRA"    , test_fillMSRA),
		]
	}
	
	//----------------------------------------------------------------------------
	// test_fillUniform
	func test_fillUniform() {
		do {
			let model = Model()
			try model.setup()
			let stream = try model.compute.requestStreams(label: "dataStream")[0]
			var data = DataView(count: 100)
			let range = -2.0...2.0
			try stream.fillUniform(data: &data, range: range, seed: 0)
			for i in 0..<data.elementCount {
				let val: Double = try data.get(at: [i])
				XCTAssert(val >= range.lowerBound && val <= range.upperBound)
			}
		} catch {
			XCTFail(String(describing: error))
		}
	}
	
	//----------------------------------------------------------------------------
	// test_fillGaussian
	func test_fillGaussian() {
		do {
			let model = Model()
			try model.setup()
			let stream = try model.compute.requestStreams(label: "dataStream")[0]
			let count = 1000
			var data = DataView(count: count)
			let mean = 10.0
			let std = 0.1
			try stream.fillGaussian(data: &data, mean: mean, std: std, seed: 0)

			let (dataMean, dataStd) = try data.meanAndStd()
			XCTAssert(abs(dataMean - mean) < 0.01)
			XCTAssert(abs(dataStd - std) < 0.01)

		} catch {
			XCTFail(String(describing: error))
		}
	}
	
	//----------------------------------------------------------------------------
	// test_fillXavier
	//  Note: because the generated numbers are random, the targetStd is just
	//        a rough approximation of what the std should be
	func test_fillXavier() {
		func validate(data: DataView, N: Double) throws {
			let buffer = try data.roReal32F()
			let mean = buffer.reduce(0, +) / Float(buffer.count)
			let std = sqrt(buffer.reduce(0, {
				let v = $1 - mean
				return $0 + Double(v * v)
			}) / Double(buffer.count))

			let targetStd = sqrt(3.0 / N) / 2

			XCTAssert(almostEquals(mean, 0, tolerance: 0.1))
			XCTAssert(almostEquals(std, targetStd, tolerance: 0.1))
		}
		
		do {
			let model = Model()
			try model.setup()
			let stream = try model.compute.requestStreams(label: "dataStream")[0]
			var data = DataView(extent: [1000, 2, 4, 5])
			
			try stream.fillXavier(data: &data, varianceNorm: .average, seed: 0)
			var N = computeVarianceNorm(shape: data.shape, varianceNorm: .average)
			try validate(data: data, N: N)
			
			try stream.fillXavier(data: &data, varianceNorm: .fanIn, seed: 0)
			N = computeVarianceNorm(shape: data.shape, varianceNorm: .fanIn)
			try validate(data: data, N: N)
			
			try stream.fillXavier(data: &data, varianceNorm: .fanOut, seed: 0)
			N = computeVarianceNorm(shape: data.shape, varianceNorm: .fanOut)
			try validate(data: data, N: N)
			
		} catch {
			XCTFail(String(describing: error))
		}
	}
	
	//----------------------------------------------------------------------------
	// test_fillMSRA
	func test_fillMSRA() {

		func validate(data: DataView, N: Double) throws {
			let buffer = try data.roReal32F()
			let mean = buffer.reduce(0, +) / Float(buffer.count)
			let std = sqrt(buffer.reduce(0, {
				let v = $1 - mean
				return $0 + Double(v * v)
			}) / Double(buffer.count))
			let targetStd = sqrt(2.0 / N)
			
			XCTAssert(almostEquals(mean, 0, tolerance: 0.1))
			XCTAssert(almostEquals(std, targetStd, tolerance: 0.1))
		}
		
		do {
			let model = Model()
			try model.setup()
			let stream = try model.compute.requestStreams(label: "dataStream")[0]
			var data = DataView(extent: [1000, 2, 4, 5])
			
			try stream.fillMSRA(data: &data, varianceNorm: .average, seed: 0)
			var N = computeVarianceNorm(shape: data.shape, varianceNorm: .average)
			try validate(data: data, N: N)
			
			try stream.fillMSRA(data: &data, varianceNorm: .fanIn, seed: 0)
			N = computeVarianceNorm(shape: data.shape, varianceNorm: .fanIn)
			try validate(data: data, N: N)
			
			try stream.fillMSRA(data: &data, varianceNorm: .fanOut, seed: 0)
			N = computeVarianceNorm(shape: data.shape, varianceNorm: .fanOut)
			try validate(data: data, N: N)
			
		} catch {
			XCTFail(String(describing: error))
		}
	}
	
} // TestFill
