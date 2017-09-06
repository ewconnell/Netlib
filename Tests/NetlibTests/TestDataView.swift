//******************************************************************************
//  Created by Edward Connell on 12/1/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
import XCTest
import Foundation

@testable import Netlib

class TestDataView: XCTestCase {
	static var allTests : [(String, (TestDataView) -> () throws -> Void)] {
		return [
			("test_learnedParameterMutation", test_learnedParameterMutation),
			("test_dataMigration"           , test_dataMigration),
			("test_mutateOnDevice"          , test_mutateOnDevice),
			("test_copyOnWriteCrossDevice"  , test_copyOnWriteCrossDevice),
			("test_copyOnWriteDevice"       , test_copyOnWriteDevice),
			("test_copyOnWrite"             , test_copyOnWrite),
			("test_columnMajorDataView"     , test_columnMajorDataView),
			("test_columnMajorStrides"      , test_columnMajorStrides),
		]
	}
	
	//----------------------------------------------------------------------------
	// test_learnedParameterMutation
	func test_learnedParameterMutation() {
		do {
			let model = Model()
			let fc = model.items.append(FullyConnected())
			try model.setup()
			let stream = try model.compute.requestStreams(label: "dataStream")[0]
			
			fc.weights.fillMethod = .indexed
			try fc.weights.setExtent([1, 1, 3, 10], using: stream)
			
			let m2 = model.copy()
			// this is done to prevent the optimizer from removing this copy
			XCTAssert(m2.log.logLevel == .error)
			
			try stream.fillXavier(data: &fc.weights.data,
			                      varianceNorm: .fanIn, seed: nil)
			XCTAssert(fc.weights.data.lastAccessMutated)
			
		} catch {
			XCTFail(String(describing: error))
		}
	}
	
	//----------------------------------------------------------------------------
	// test_dataMigration
	func test_dataMigration() {
		do {
			let model = Model()
			try model.setup()
			let streams = try model.compute.requestStreams(label: "dataStream", deviceIds: [0, 1])
			let multiDevice = streams[0].device.id != streams[1].device.id
			
			// this test needs 2 devices
			if streams.count != 2 { return }
			
			var data = DataView(rows: 4, cols: 3)
			_ = try data.rwReal8U()
			XCTAssert(!data.dataArray.lastAccessCopiedBuffer)
			
			_ = try data.roReal8U()
			XCTAssert(!data.dataArray.lastAccessCopiedBuffer)
			
			_ = try data.ro(using: streams[0])
			XCTAssert(data.dataArray.lastAccessCopiedBuffer)
			
			_ = try data.roReal8U()
			XCTAssert(!data.dataArray.lastAccessCopiedBuffer)
			
			_ = try data.rw(using: streams[0])
			XCTAssert(!data.dataArray.lastAccessCopiedBuffer)

			if multiDevice {
				_ = try data.ro(using: streams[1])
				XCTAssert(data.dataArray.lastAccessCopiedBuffer)
			}

			_ = try data.ro(using: streams[0])
			XCTAssert(!data.dataArray.lastAccessCopiedBuffer)
			
			_ = try data.ro(using: streams[1])
			XCTAssert(!data.dataArray.lastAccessCopiedBuffer)
			
			_ = try data.rw(using: streams[0])
			XCTAssert(!data.dataArray.lastAccessCopiedBuffer)

			if multiDevice {
				_ = try data.ro(using: streams[1])
				XCTAssert(data.dataArray.lastAccessCopiedBuffer)
			}

			_ = try data.rw(using: streams[1])
			XCTAssert(!data.dataArray.lastAccessCopiedBuffer)

			if multiDevice {
				_ = try data.rw(using: streams[0])
				XCTAssert(data.dataArray.lastAccessCopiedBuffer)

				_ = try data.rw(using: streams[1])
				XCTAssert(data.dataArray.lastAccessCopiedBuffer)
			}

			_ = try data.roReal8U()
			XCTAssert(data.dataArray.lastAccessCopiedBuffer)
			
		} catch {
			XCTFail(String(describing: error))
		}
	}
	
	//----------------------------------------------------------------------------
	// test_mutateOnDevice
	func test_mutateOnDevice() {
		do {
			let model = Model()
			try model.setup()
			let streams = try model.compute.requestStreams(label: "dataStream", deviceIds: [0, 1])
			
			var data0 = DataView(rows: 3, cols: 2)
			try streams[0].fillWithIndex(data: &data0, startingAt: 0)

			let value1: Float = try data0.get(at: [1, 1])
			XCTAssert(value1 == 3.0)
			
			// migrate the data to the devices
			_ = try data0.ro(using: streams[0])
			
			// sum device 0 copy should be 15
			var sum = DataView(count: 1)
			try streams[0].asum(x: data0.flattened(), result: &sum)
			var sumValue: Float = try sum.get()
			XCTAssert(sumValue == 15.0)
			
			let data1 = data0
			_ = try data1.ro(using: streams[1])
			
			// sum device 1 copy should be 15
			try streams[1].asum(x: data0.flattened(), result: &sum)
			sumValue = try sum.get()
			XCTAssert(sumValue == 15.0)
			
			// clear stream 0 copy
			try streams[0].fill(data: &data0, with: 0)
			
			// sum device 1 copy should still be 15
			try streams[1].asum(x: data1.flattened(), result: &sum)
			sumValue = try sum.get()
			XCTAssert(sumValue == 15.0)
			//			print(sumValue)
			
		} catch {
			XCTFail(String(describing: error))
		}
	}
	
	//----------------------------------------------------------------------------
	// test_copyOnWriteDevice
	func test_copyOnWriteDevice() {
		do {
			let model = Model()
			try model.setup()
			let stream = try model.compute.requestStreams(label: "dataStream")[0]
			
			let testIndex = [1, 1]
			var data1 = DataView(rows: 3, cols: 2)
			try cpuFillWithIndex(data: &data1, startingAt: 0)
			let value1: Float = try data1.get(at: testIndex)
			XCTAssert(value1 == 3.0)
			
			// migrate the data to the device
			_ = try data1.ro(using: stream)
			
			// copy and mutate data
			// the data will be duplicated wherever the source is
			var data2 = data1
			let value2: Float = try data2.get(at: testIndex)
			XCTAssert(value2 == 3.0)
			try data2.set(value: 7, at: [1, 1])
			
			let value1a: Float = try data1.get(at: testIndex)
			XCTAssert(value1a == 3.0)
			
			let value2a: Float = try data2.get(at: testIndex)
			XCTAssert(value2a == 7.0)
		} catch {
			XCTFail(String(describing: error))
		}
	}
	
	//----------------------------------------------------------------------------
	// test_copyOnWriteCrossDevice
	func test_copyOnWriteCrossDevice() {
		do {
			let model = Model()
			try model.setup()
			let streams = try model.compute.requestStreams(label: "dataStream", deviceIds: [0, 1])
			let multiDevice = streams[0].device.id != streams[1].device.id

			// don't test unless we have multiple devices
			if !multiDevice { return }

			let testIndex = [0, 0, 1, 1]
			var data1 = DataView(rows: 3, cols: 2)
			try streams[0].fillWithIndex(data: &data1, startingAt: 0)
			let value1: Float = try data1.get(at: testIndex)
			XCTAssert(value1 == 3.0)
			
			// migrate the data to the devices
			_ = try data1.ro(using: streams[0])
			_ = try data1.ro(using: streams[1])
			
			// sum device 0 copy should be 15
			var sum = DataView(count: 1)
			try streams[0].asum(x: data1.flattened(), result: &sum)
			var sumValue: Float = try sum.get()
			XCTAssert(sumValue == 15.0)
			
			// clear the device 0 master copy
			try streams[0].fill(data: &data1, with: 0)
			
			// sum device 1 copy should now also be 0
			try streams[1].asum(x: data1.flattened(), result: &sum)
			sumValue = try sum.get()
			XCTAssert(sumValue == 0.0)
		} catch {
			XCTFail(String(describing: error))
		}
	}
	
	//----------------------------------------------------------------------------
	// test_copyOnWrite
	func test_copyOnWrite() {
		do {
			let testIndex = [1, 1]
			var data1 = DataView(rows: 3, cols: 2)
			try cpuFillWithIndex(data: &data1, startingAt: 0)
			let value1: Float = try data1.get(at: testIndex)
			XCTAssert(value1 == 3.0)
			
			var data2 = data1
			let value2: Float = try data2.get(at: testIndex)
			XCTAssert(value2 == 3.0)
			try data2.set(value: 7, at: testIndex)
			
			let value1a: Float = try data1.get(at: testIndex)
			XCTAssert(value1a == 3.0)
			
			let value2a: Float = try data2.get(at: testIndex)
			XCTAssert(value2a == 7.0)
		} catch {
			XCTFail(String(describing: error))
		}
	}
	
	//----------------------------------------------------------------------------
	// test_columnMajorStrides
	func test_columnMajorStrides() {
		let extent_nchw = [1, 2, 3, 4]
		let rmShape4 = Shape(extent: extent_nchw)
		XCTAssert(rmShape4.strides == [24, 12, 4, 1])
		
		let cmShape4 = Shape(extent: extent_nchw, colMajor: true)
		XCTAssert(cmShape4.strides == [24, 12, 1, 3])

		let extent_nhwc = [1, 3, 4, 2]
		let irmShape4 = Shape(extent: extent_nhwc, layout: .nhwc)
		XCTAssert(irmShape4.strides == [24, 8, 2, 1])
		
		let icmShape4 = Shape(extent: extent_nhwc, layout: .nhwc, colMajor: true)
		XCTAssert(icmShape4.strides == [24, 2, 6, 1])
		
		//-------------------------------------
		// rank 5
//		let extent5  = Extent(10, 3, 4, 2, 3)
//		let rmShape5 = Shape(extent: extent5)
//		XCTAssert(rmShape5.strides[0] == 72)
//		XCTAssert(rmShape5.strides[1] == 24)
//		XCTAssert(rmShape5.strides[2] == 6)
//		XCTAssert(rmShape5.strides[3] == 3)
//		XCTAssert(rmShape5.strides[4] == 1)
//		
//		// rank 5
//		let irmShape5 = Shape(extent: extent5, isInterleaved: true)
//		XCTAssert(irmShape5.strides[0] == 72)
//		XCTAssert(irmShape5.strides[1] == 1)
//		XCTAssert(irmShape5.strides[2] == 18)
//		XCTAssert(irmShape5.strides[3] == 9)
//		XCTAssert(irmShape5.strides[4] == 3)
//		
//		// rank 5
//		let cmShape5 = Shape(extent: extent5, isColMajor: true)
//		XCTAssert(cmShape5.strides[0] == 72)
//		XCTAssert(cmShape5.strides[1] == 24)
//		XCTAssert(cmShape5.strides[2] == 6)
//		XCTAssert(cmShape5.strides[3] == 1)
//		XCTAssert(cmShape5.strides[4] == 2)
//		
//		// rank 5
//		let icmShape5 = Shape(extent: extent5, isInterleaved: true, isColMajor: true)
//		XCTAssert(icmShape5.strides[0] == 72)
//		XCTAssert(icmShape5.strides[1] == 1)
//		XCTAssert(icmShape5.strides[2] == 18)
//		XCTAssert(icmShape5.strides[3] == 3)
//		XCTAssert(icmShape5.strides[4] == 6)
	}
	
	//----------------------------------------------------------------------------
	// test_columnMajorDataView
	func test_columnMajorDataView() {
		do {
			// load linear buffer with values in col major order
			let cmArray: [UInt8] = [0, 3, 1, 4, 2, 5]
			let extent  = [2, 3]
			let cmShape = Shape(extent: extent, colMajor: true)
			
			// create a data view
			var cmData   = DataView(shape: cmShape, dataType: .real8U)
			let cmBuffer = try cmData.rwReal8U()
			for i in 0..<cmArray.count { cmBuffer[i] = cmArray[i] }
			
			// test col major indexing
			var i: UInt8 = 0
			for row in 0..<cmData.rows {
				for col in 0..<cmData.cols {
					let value: UInt8 = try cmData.get(at: [row, col])
					XCTAssert(value == i)
					i += 1
				}
			}
			
			// create row major view from cmData, this will copy and reorder
			let rmShape = Shape(extent: extent)
			let rmData = try DataView(from: cmData, asShape: rmShape)
			let rmBuffer = try rmData.roReal8U()
			for i in 0..<rmData.elementCount { XCTAssert(rmBuffer[i] == UInt8(i))	}
			
			// test row major indexing
			i = 0
			for row in 0..<rmData.rows {
				for col in 0..<rmData.cols {
					let value: UInt8 = try rmData.get(at: [row, col])
					XCTAssert(value == i)
					i += 1
				}
			}
		} catch {
			XCTFail(String(describing: error))
		}
	}
}
