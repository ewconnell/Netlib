//******************************************************************************
//  Created by Edward Connell on 6/7/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
import XCTest
import Foundation

@testable import Netlib

class TestProvider: XCTestCase
{
	static var allTests : [(String, (TestProvider) -> () throws -> Void)] {
		return [
			("test_LmdbProviderCreate", test_LmdbProviderCreate),
			("test_LmdbProviderWrite" , test_LmdbProviderWrite),
			("test_LmdbProviderRead"  , test_LmdbProviderRead),
		]
	}
	
	//----------------------------------------------------------------------------
	// test_LmdbProviderCreate
	func test_LmdbProviderCreate() {
		do {
			// create and add some data
			let sourceData: [UInt8] = [1, 2, 3]
			var key: DataKey

			do {
				let lmdb = LmdbProvider()
				lmdb.connection = "mnist_train"

				try lmdb.removeDatabase()
				let session = try lmdb.open(mode: .readWrite)
				let t = try session.dataTable.transaction(parent: nil, mode: .readWrite)
				key = try t.append(data: sourceData)
				try t.commit()
			}

			// now open for reading
			do {
				let lmdb = LmdbProvider()
				lmdb.connection = "mnist_train"

				let session = try lmdb.open(mode: .readOnly)
				let cursor = try session.dataTable.transaction().cursor()
				if let record = try cursor.first() {
					for (i, value) in record.data.enumerated() {
						XCTAssert(sourceData[i] == value)
					}
				}

				if let record = try cursor.set(key: key) {
					for (i, value) in record.data.enumerated() {
						XCTAssert(sourceData[i] == value)
					}
				}
			}

		} catch {
			print(error)
			XCTFail()
		}
	}
	
	//----------------------------------------------------------------------------
	// test_LmdbProviderWrite
	func test_LmdbProviderWrite() {
		self.measure {
			do {
				// create and add some data
				let sourceData = [UInt8](repeatElement(0, count: 400))
				let lmdb = LmdbProvider()
				lmdb.connection = "test_LmdbProviderWrite"
				
				try lmdb.removeDatabase()
				let session = try lmdb.open(mode: .readWrite)
				
				for _ in 0..<10 {
					let txnGroup = try session.transaction()
					let txnData = try session.dataTable.transaction(parent: txnGroup, mode: .readWrite)
					
					for _ in 0..<1000 {
						_ = try txnData.append(data: sourceData)
					}
					
					try txnData.commit()
					try txnGroup.commit()
				}

			} catch {
				print(error)
				XCTFail()
			}
		}
	}

	//----------------------------------------------------------------------------
	// test_LmdbProviderRead
	func test_LmdbProviderRead() {
		do {
			// create and add some data
			do {
				let sourceData: [UInt8] = [1, 2, 3]
				
				let lmdb = LmdbProvider()
				lmdb.connection = "mnist_train"
				try lmdb.removeDatabase()
				let session = try lmdb.open(mode: .readWrite)
				
				let t2 = try session.dataTable.transaction(mode: .readWrite)
				for i in 0..<9 {
					var data2 = sourceData
					data2[0] = UInt8(i)
					_ = try t2.append(data: data2)
				}
				try t2.commit()
			}

			// now do some reads
			let lmdb = LmdbProvider()
			lmdb.connection = "mnist_train"
			let session = try lmdb.open(mode: .readOnly)
			
			for _ in 0..<200 {
				let cursor = try session.dataTable.transaction().cursor()
				var counter: UInt8 = 0
				while let record = try cursor.next() {
					XCTAssert(record.data[0] == counter)
					counter += 1
				}
			}
		} catch {
			print(error)
			XCTFail()
		}
	}
}
