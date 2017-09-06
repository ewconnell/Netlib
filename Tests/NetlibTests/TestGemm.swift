//******************************************************************************
//  Created by Edward Connell on 12/6/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
import XCTest
import Foundation

@testable import Netlib

class TestGemm: XCTestCase {
	static var allTests : [(String, (TestGemm) -> () throws -> Void)] {
		return [
			("test_gemmNoTrans"         , test_gemmNoTrans),
			("test_gemmTrans"           , test_gemmTrans),
			("test_gemmSubRegionNoTrans", test_gemmSubRegionNoTrans),
			("test_gemmSubRegionTrans"  , test_gemmSubRegionTrans),
		]
	}
	
	//----------------------------------------------------------------------------
	// test_gemmNoTrans
	func test_gemmNoTrans() {
		do {
			let model = Model()
			try model.setup()
			let stream = try model.compute.requestStreams(label: "dataStream")[0]
			
			var A = DataView(rows: 3, cols: 2)
			var B = DataView(rows: 2, cols: 3)
			var C = DataView(rows: 3, cols: 3)
			try stream.fillWithIndex(data: &A, startingAt: 1)
			try stream.fillWithIndex(data: &B, startingAt: 1)
			try stream.fill(data: &C, with: 0)
			
			try stream.gemm(alpha: 1, transA: .noTranspose, matrixA: A,
			                transB: .noTranspose, matrixB: B,
			                beta: 0, matrixC: &C)
			
			let expectedC: [[Float]] = [
				[ 9, 12, 15],
				[19, 26, 33],
				[29, 40, 51]
			]
			
			for r in 0..<C.rows {
				for c in 0..<C.cols {
					let value: Float = try C.get(at: [r, c])
					XCTAssert(almostEquals(value, expectedC[r][c]))
				}
			}
			
		} catch {
			XCTFail(String(describing: error))
		}
	}
	
	//----------------------------------------------------------------------------
	// test_gemmTrans
	func test_gemmTrans() {
		do {
			let model = Model()
			try model.setup()
			let stream = try model.compute.requestStreams(label: "dataStream")[0]
			
			var A = DataView(rows: 2, cols: 3)
			var B = DataView(rows: 3, cols: 2)
			var C = DataView(rows: 3, cols: 3)
			try stream.fillWithIndex(data: &A, startingAt: 1)
			try stream.fillWithIndex(data: &B, startingAt: 1)
			try stream.fill(data: &C, with: 0)
			
			try stream.gemm(alpha: 1, transA: .transpose, matrixA: A,
			                transB: .transpose, matrixB: B,
			                beta: 0, matrixC: &C)
			
			let expectedC: [[Float]] = [
				[ 9, 19, 29],
				[12, 26, 40],
				[15, 33, 51]
			]
			
			for r in 0..<C.rows {
				for c in 0..<C.cols {
					let value: Float = try C.get(at: [r, c])
					XCTAssert(almostEquals(value, expectedC[r][c]))
				}
			}
			
		} catch {
			XCTFail(String(describing: error))
		}
	}
	
	//----------------------------------------------------------------------------
	// test_gemmSubRegionNoTrans
	func test_gemmSubRegionNoTrans() {
		do {
			let model = Model()
			try model.setup()
			let stream = try model.compute.requestStreams(label: "dataStream")[0]
			
			var A = DataView(rows: 5, cols: 4)
			var B = DataView(rows: 4, cols: 5)
			var C = DataView(rows: 5, cols: 5)
			try stream.fill(data: &A, with: 0)
			try stream.fill(data: &B, with: 0)
			try stream.fill(data: &C, with: 0)
			
			var subA = try A.referenceView(offset: [1, 1], extent: [3, 2], using: stream)
			try stream.fillWithIndex(data: &subA, startingAt: 1)
			
			var subB = try B.referenceView(offset: [1, 1], extent: [2, 3], using: stream)
			try stream.fillWithIndex(data: &subB, startingAt: 1)
			
			var subC = try C.referenceView(offset: [1, 1], extent: [3, 3], using: stream)
			
			try stream.gemm(alpha: 1, transA: .noTranspose, matrixA: subA,
			                transB: .noTranspose, matrixB: subB,
			                beta: 0, matrixC: &subC)
			
			let expectedC: [[Float]] = [
				[ 9, 12, 15],
				[19, 26, 33],
				[29, 40, 51]
			]
			
			for r in 0..<subC.rows {
				for c in 0..<subC.cols {
					let value: Float = try subC.get(at: [r, c])
					XCTAssert(almostEquals(value, expectedC[r][c]))
				}
			}
			
		} catch {
			XCTFail(String(describing: error))
		}
	}
	
	//----------------------------------------------------------------------------
	// test_gemmSubRegionTrans
	func test_gemmSubRegionTrans() {
		do {
			let model = Model()
			try model.setup()
			let stream = try model.compute.requestStreams(label: "dataStream")[0]
			
			var A = DataView(rows: 3, cols: 5)
			var B = DataView(rows: 4, cols: 4)
			var C = DataView(rows: 4, cols: 5)
			try stream.fill(data: &A, with: 0)
			try stream.fill(data: &B, with: 0)
			try stream.fill(data: &C, with: 0)
			
			var subA = try A.referenceView(offset: [1, 1], extent: [2, 3], using: stream)
			try stream.fillWithIndex(data: &subA, startingAt: 1)
			
			var subB = try B.referenceView(offset: [1, 1], extent: [3, 2], using: stream)
			try stream.fillWithIndex(data: &subB, startingAt: 1)
			
			var subC = try C.referenceView(offset: [1, 1], extent: [3, 3], using: stream)
			
			try stream.gemm(alpha: 1, transA: .transpose, matrixA: subA,
			                transB: .transpose, matrixB: subB,
			                beta: 0, matrixC: &subC)
			
			let expectedC: [[Float]] = [
				[ 9, 19, 29],
				[12, 26, 40],
				[15, 33, 51]
			]
			
			for r in 0..<subC.rows {
				for c in 0..<subC.cols {
					let value: Float = try subC.get(at: [r, c])
					XCTAssert(almostEquals(value, expectedC[r][c]))
				}
			}
			
		} catch {
			XCTFail(String(describing: error))
		}
	}
}
