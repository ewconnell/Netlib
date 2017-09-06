//******************************************************************************
//  Created by Edward Connell on 9/27/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
import XCTest
import Foundation

@testable import Netlib

class TestDatabase: XCTestCase {
	static var allTests : [(String, (TestDatabase) -> () throws -> Void)] {
		return [
			("test_dataCodec"           , test_dataCodec),
			("test_indexedConversion4_1", test_indexedConversion4_1),
			("test_indexedConversion4"  , test_indexedConversion4),
			("test_rgbaConversion"      , test_rgbaConversion),
			("test_rgbConversion"       , test_rgbConversion),
			("test_uniformFolderLoad"   , test_uniformFolderLoad),
			("test_nonUniformFolderLoad", test_nonUniformFolderLoad),
		]
	}
	
	//----------------------------------------------------------------------------
	// This creates an indexed buffer, encodes and decodes as unstructured data
	func test_dataCodec() {
		do {
			let model = Model()
			model.concurrencyMode = .serial
			
			// source is row major samples
			let source = TestDataSource()
			source.itemCount = 2
			source.itemExtent = [1, 3, 5, 4]
			source.dataLayout = .nhwc
			source.indexBy = .values
			source.containerTemplate.codecType = .data
			source.normalizeIndex = false
			source.dataType = .real8U
			
			// build database
			let database = model.items.append(Database())
			defer { try? database.provider.removeDatabase() }
			database.connection = "test_dataCodec"
			database.source = source
			database.rebuild = .always
			database.deviceIds = [0]
			database.dataLayout = source.dataLayout
			database.dataType = .real32F
			try model.setup()
			
			// get data
			let selection = Selection(count: source.itemCount)
			let _ = try database.forward(mode: .inference, selection: selection)
			
			// validate output
			var data = DataView()
			try database.copyOutput(connectorName: "data", to: &data)
			
			let sourceShape = Shape(items: source.itemCount,
			                        shape: Shape(extent: source.itemExtent!,
			                                     layout: source.dataLayout))
			XCTAssert(data.shape == sourceShape)
			
			itemLoop: for item in 0..<data.shape.items {
				// the source data type is UInt8 indexes, so use the same type
				// for comparison. The database output is Float, so the index values
				// will be normalized
				var i: UInt8 = 0
				for row in 0..<data.shape.rows {
					for col in 0..<data.shape.cols {
						for chan in 0..<data.shape.channels {
							let index = data.shape.index(item: item, channel: chan, row: row, col: col)
							let value: Float = try data.get(at: index)
							if !almostEquals(value, Float(i)) {
								XCTFail("values don't match")
								break itemLoop
							}
						}
						i += 1
					}
				}
			}
		} catch {
			XCTFail(String(describing: error))
		}
	}
	
	//----------------------------------------------------------------------------
	// This generates interleaved RGBA data, encodes in png, stores,
	// decodes, and returns data as planar Float
	func test_indexedConversion4_1() {
		do {
			let model = Model()
			//			model.concurrencyMode = .serial
			
			// store data with png compression, make it a default on the model
			let format = ImageFormat { $0.encoding = .png; $0.channelFormat = .gray }
			model.defaults = [Default { $0.property = "ImageCodec.format"; $0.object = format }]
			
			// source RGBA<UInt8> is 0123 interleaved
			let source = TestDataSource {
				$0.itemCount = 2
				$0.itemExtent = [1, 3, 5, 4]
				$0.channelFormat = .rgba
				$0.dataLayout = .nhwc
				$0.indexBy = .channels
				$0.containerTemplate.codecType = .image
				$0.normalizeIndex = true
			}

			// build database
			let database = model.items.append(Database())
			defer { try? database.provider.removeDatabase() }
			database.connection = "test_indexedConversion4_1"
			database.source = source
			database.rebuild = .always
			database.deviceIds = [0]
			try model.setup()
			
			// get data as .real32F
			database.dataType = .real32F
			let selection = Selection(count: source.itemCount)
			let _ = try database.forward(mode: .inference, selection: selection)
			
			// validate output
			let data = database.outputs["data"]!.items[0].data
			let data8 = try DataView(from: data, asDataType: .real8U,
			                         normalizeInts: true)

			// gray scale is always converted to .nchw by the codec
			let expectedExtent =
				[source.itemCount, 1, source.itemExtent![1], source.itemExtent![2]]
			XCTAssert(data8.extent == expectedExtent)
			
			let flatData = try data8.flattened().roReal8U()
			for value in flatData {
				if value != 124 {
					XCTFail("values don't match")
				}
			}
		} catch {
			XCTFail(String(describing: error))
		}
	}
	
	//----------------------------------------------------------------------------
	// This generates rgba pixel data, encodes in png, stores,
	// decodes, and returns data as planar Float
	func test_indexedConversion4() {
		do {
			let model = Model()
			model.concurrencyMode = .serial

			// source RGBA<UInt8> is 0123 interleaved
			let source = TestDataSource {
				$0.itemCount = 2
				$0.itemExtent = [1, 3, 5, 4]
				$0.channelFormat = .rgba
				$0.dataLayout = .nhwc
				$0.indexBy = .channels
				$0.containerTemplate.codecType = .image
				$0.normalizeIndex = true
			}

			// store data with png compression, make it a default on the model
			let format = ImageFormat { $0.encoding = .png; $0.channelFormat = .rgba }
			model.defaults = [Default { $0.property = "ImageCodec.format"; $0.object = format }]

			// build database
			let database = model.items.append(Database())
			defer { try? database.provider.removeDatabase() }
			database.connection = "test_indexedConversion4"
			database.source = source
			database.dataLayout = .nchw
			database.rebuild = .always
			database.deviceIds = [0]
			try model.setup()

			// get data as .real32F planar
			database.dataType = .real32F
			let selection = Selection(count: source.itemCount)
			let _ = try database.forward(mode: .inference, selection: selection)

			// validate output
			let data = database.outputs["data"]!.items[0].data
			XCTAssert(data.extent ==
				data.makeExtent(items: source.itemCount, channels: 4, rows: 3, cols: 5))

			let channelExtent = data.shape.makeExtent(items: 1, channels: 1,
			                                          rows: data.rows, cols: data.cols)
			for item in 0..<data.items {
				for chan in 0..<data.channels {
					let index = data.index(item: item, channel: chan, row: 0, col: 0)
					let view = data.view(offset: index, extent: channelExtent)
					let flatView = view.flattened()
					let buffer = try flatView.roReal32F()
					let expectedValue = Float(norm: UInt8(norm: Double(chan + 1) / Double(data.channels)))
					for value in buffer {
						if value != expectedValue {
							XCTFail("values don't match")
						}
					}
				}
			}

		} catch {
			XCTFail(String(describing: error))
		}
	}
	
	//----------------------------------------------------------------------------
	func test_rgbaConversion() {
		do {
			guard let file1 = getBundle(for: type(of: self)).path(
				forResource: "samples/images/cat_gray_8", ofType: "png")
				else { XCTFail(); return }
			
			guard let file2 = getBundle(for: type(of: self)).path(
				forResource: "samples/images/cat_gray_8", ofType: "jpg")
				else { XCTFail(); return }
			
			guard let file3 = getBundle(for: type(of: self)).path(
				forResource: "samples/images/cat_RGB_8", ofType: "png")
				else { XCTFail(); return }
			
			guard let file4 = getBundle(for: type(of: self)).path(
				forResource: "samples/images/cat_RGB_8", ofType: "jpg")
				else { XCTFail(); return }
			
			let model = Model()
			
			// set output format
			let format = ImageFormat { $0.encoding = .png; $0.channelFormat = .rgba }
			model.defaults = [Default { $0.property = "ImageCodec.format"; $0.object = format }]
			
			let fileList = FileList()
			fileList.items = [
				DataContainer(contentsOf: Uri(filePath: file1), codecType: .image),
				DataContainer(contentsOf: Uri(filePath: file2), codecType: .image),
				DataContainer(contentsOf: Uri(filePath: file3), codecType: .image),
				DataContainer(contentsOf: Uri(filePath: file4), codecType: .image),
			]
			
			let database = model.items.append(Database())
			defer { try? database.provider.removeDatabase() }
			database.connection = "test_rgbaConversion"
			database.source = fileList
			database.rebuild = .always
			try model.setup()
			
		} catch {
			XCTFail(String(describing: error))
		}
	}
	
	//----------------------------------------------------------------------------
	func test_rgbConversion() {
		do {
			guard let file1 = getBundle(for: type(of: self)).path(
				forResource: "samples/images/cat_gray_8", ofType: "png")
				else { XCTFail(); return }
			
			guard let file2 = getBundle(for: type(of: self)).path(
				forResource: "samples/images/cat_gray_8", ofType: "jpg")
				else { XCTFail(); return }
			
			guard let file3 = getBundle(for: type(of: self)).path(
				forResource: "samples/images/cat_RGB_8", ofType: "png")
				else { XCTFail(); return }
			
			guard let file4 = getBundle(for: type(of: self)).path(
				forResource: "samples/images/cat_RGB_8", ofType: "jpg")
				else { XCTFail(); return }
			
			let model = Model()
			
			// set output format
			let format = ImageFormat { $0.encoding = .jpeg; $0.channelFormat = .rgb }
			model.defaults = [Default { $0.property = "ImageCodec.format"; $0.object = format }]
			
			let fileList = FileList()
			fileList.items = [
				DataContainer(contentsOf: Uri(filePath: file1), codecType: .image),
				DataContainer(contentsOf: Uri(filePath: file2), codecType: .image),
				DataContainer(contentsOf: Uri(filePath: file3), codecType: .image),
				DataContainer(contentsOf: Uri(filePath: file4), codecType: .image),
			]
			
			let database = model.items.append(Database())
			defer { try? database.provider.removeDatabase() }
			database.connection = "test_rgbConversion"
			database.source = fileList
			database.rebuild = .always
			try model.setup()
			
		} catch {
			XCTFail(String(describing: error))
		}
	}
	
	//----------------------------------------------------------------------------
	func test_uniformFolderLoad() {
		do {
			guard let file1 = getBundle(for: type(of: self)).path(
				forResource: "samples/images/cat_gray_8", ofType: "png")
				else { XCTFail(); return }
			
			guard let file2 = getBundle(for: type(of: self)).path(
				forResource: "samples/images/example1_8", ofType: "png")
				else { XCTFail(); return }
			
			guard let file3 = getBundle(for: type(of: self)).path(
				forResource: "samples/images/cat_gray_8", ofType: "jpg")
				else { XCTFail(); return }
			
			let model = Model()
			//			model.concurrencyMode = .serial
			let fileList = FileList()
			fileList.items = [
				DataContainer(contentsOf: Uri(filePath: file1), codecType: .image),
				DataContainer(contentsOf: Uri(filePath: file2), codecType: .image),
				DataContainer(contentsOf: Uri(filePath: file3), codecType: .image),
			]
			
			let database = model.items.append(Database())
			defer { try? database.provider.removeDatabase() }
			database.connection = "test_uniformFolderLoad"
			database.source = fileList
			database.rebuild = .always
			database.requireUniform = .numberOfChannels
			database.streamOutput = true
			try model.setup()
			
		} catch {
			XCTFail(String(describing: error))
		}
	}
	
	//----------------------------------------------------------------------------
	func test_nonUniformFolderLoad() {
		var exceptionThrown = false
		do {
			let model = Model()
			model.log.silent = true
			//			model.concurrencyMode = .serial
			let fileList = FileList()
			fileList.directory =
				getBundle(for: type(of: self)).resourcePath! + "/samples/images"
			
			fileList.filter = ["png"]
			let database = model.items.append(Database())
			defer { try? database.provider.removeDatabase() }
			database.connection = "test_nonUniformFolderLoad"
			database.source = fileList
			database.rebuild = .always
			database.requireUniform = .numberOfChannels
			try model.setup()
			
		} catch {
			// non uniform data set should throw an error
			exceptionThrown = true
		}
		if !exceptionThrown { XCTFail("exception should have been thrown") }
	}
}
