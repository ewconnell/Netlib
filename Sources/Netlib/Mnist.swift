//******************************************************************************
//  Created by Edward Connell on 5/25/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
import Foundation
import Dispatch

final public class Mnist : DataSourceBase, DataSource, InitHelper {
	// initializers
	public required init() {
		trainingImages =
			Uri(string: "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
				  cacheFile: "train-images-idx3-ubyte.gz")

		trainingLabels =
			Uri(string: "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
				  cacheFile: "train-labels-idx1-ubyte.gz")

		validationImages =
			Uri(string: "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
				  cacheFile: "t10k-images-idx3-ubyte.gz")

		validationLabels =
			Uri(string: "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
				  cacheFile: "t10k-labels-idx1-ubyte.gz")

		super.init()
		containerTemplate.codecType = .image
	}

	//----------------------------------------------------------------------------
	// DataSet
	public enum DataSet : String, EnumerableType { case training, validation }

	//----------------------------------------------------------------------------
	// properties
	public var containerTemplate = DataContainer() { didSet{onSet("containerTemplate")} }
	public var dataSet: DataSet?                   { didSet{onSet("dataSet")} }
	public var maxCount = Int.max                  { didSet{onSet("maxCount")} }
	public var trainingImages: Uri				         { didSet{onSet("trainingImages")} }
	public var trainingLabels: Uri	               { didSet{onSet("trainingLabels")} }
	public var validationImages: Uri			         { didSet{onSet("validationImages")} }
	public var validationLabels: Uri	             { didSet{onSet("validationLabels")} }

	// count
	public var count: Int {
		return min(imagesHeader?.numImages ?? 0, maxCount)
	}

	// local data
	private var imagesData: [UInt8]!
	private var labelsData: [UInt8]!
	private var imagesHeader: ImagesHeader!
	private var shape: Shape!

	//----------------------------------------------------------------------------
	// addAccessors
	public override func addAccessors() {
		super.addAccessors()
		addAccessor(name: "containerTemplate",
		            get: { [unowned self] in self.containerTemplate },
		            set: { [unowned self] in self.containerTemplate = $0 })
		addAccessor(name: "dataSet",
		            get: { [unowned self] in self.dataSet },
		            set: { [unowned self] in self.dataSet = $0 })
		addAccessor(name: "maxCount",
		            get: { [unowned self] in self.maxCount },
		            set: { [unowned self] in self.maxCount = $0 })
		addAccessor(name: "trainingImages",
		            get: { [unowned self] in self.trainingImages },
		            set: { [unowned self] in self.trainingImages = $0 })
		addAccessor(name: "trainingLabels",
		            get: { [unowned self] in self.trainingLabels },
		            set: { [unowned self] in self.trainingLabels = $0 })
		addAccessor(name: "validationImages",
		            get: { [unowned self] in self.validationImages },
		            set: { [unowned self] in self.validationImages = $0 })
		addAccessor(name: "validationLabels",
		            get: { [unowned self] in self.validationLabels },
		            set: { [unowned self] in self.validationLabels = $0 })
	}

	//----------------------------------------------------------------------------
	// setupData
	public func setupData(taskGroup: TaskGroup) throws {
		// validate
		guard let dataSet = dataSet else {
			writeLog("\(namespaceName): dataSet must be specified")
			throw ModelError.setupFailed
		}

		// select set
		let imagesUri = dataSet == .training ? trainingImages : validationImages
		let labelsUri = dataSet == .training ? trainingLabels : validationLabels

		// create work group for dependents and download the data
		let group = WorkGroup(taskGroup, DispatchGroup())
		try DownloadTask.download(uri: imagesUri, group: group) {self.imagesData = $0}
		try DownloadTask.download(uri: labelsUri, group: group) {self.labelsData = $0}
		group.dispatchGroup!.wait()

		//-----------------------------
		// validate
		imagesHeader = imagesData.withUnsafeBufferPointer {
			$0.baseAddress!.withMemoryRebound(to: RawImagesHeader.self, capacity: 1) {
				ImagesHeader(raw: $0[0])
			}
		}

		guard imagesHeader.magicNumber == 2051 else {
			writeLog("\(namespaceName): Images data has invalid magic number")
			throw ModelError.setupFailed
		}
		
		// channels(1) x rows(28) x cols(28)
		shape = Shape(extent: [1, 1, imagesHeader.rows, imagesHeader.cols],
		              channelFormat: .gray)
		
		let labelsHeader = labelsData.withUnsafeBufferPointer {
			$0.baseAddress!.withMemoryRebound(to: RawLabelsHeader.self, capacity: 1) {
				LabelsHeader(raw: $0[0])
			}
		}
		
		guard labelsHeader.magicNumber == 2049 else {
			writeLog("\(namespaceName): Labels data has invalid magic number")
			throw ModelError.setupFailed
		}
		
		guard labelsHeader.numImages == imagesHeader.numImages else {
			writeLog("\(namespaceName): Number of images (\(self.imagesHeader.numImages)" +
				" and labels (\(labelsHeader.numImages)) do not match")
			throw ModelError.setupFailed
		}
	}
	
	//----------------------------------------------------------------------------
	// getItem
	public func getItem(at index: Int, taskGroup: TaskGroup) throws -> ModelLabeledContainer {
		assert(index < count, "Index out of range")
		let size = shape.elementCount
		let imageIndex = MemoryLayout<RawImagesHeader>.size + (index * size)
		let labelIndex = MemoryLayout<RawLabelsHeader>.size + index

		// create data object
		let data = imagesData.withUnsafeBufferPointer {
			return DataView(shape: shape, dataType: .real8U,
				start: $0.baseAddress!.advanced(by: imageIndex), count: size)
		}
		
		return try containerTemplate.copyTemplate(index: index, taskGroup: taskGroup) {
			$0.data = data
			$0.labelValue = [Double(labelsData[labelIndex])]
		}
	}
}

//------------------------------------------------------------------------------
// types
private struct RawImagesHeader {
	let magicNumber: Int32
	let numImages: Int32
	let rows: Int32
	let cols: Int32
}

private struct ImagesHeader {
	init(raw: RawImagesHeader) {
		magicNumber = Int(Int32(bigEndian: raw.magicNumber))
		numImages = Int(Int32(bigEndian: raw.numImages))
		rows = Int(Int32(bigEndian: raw.rows))
		cols = Int(Int32(bigEndian: raw.cols))
	}
	let magicNumber: Int
	let numImages: Int
	let rows: Int
	let cols: Int
}

private struct RawLabelsHeader {
	let magicNumber: Int32
	let numImages: Int32
}

private struct LabelsHeader {
	init(raw: RawLabelsHeader) {
		magicNumber = Int(Int32(bigEndian: raw.magicNumber))
		numImages = Int(Int32(bigEndian: raw.numImages))
	}
	let magicNumber: Int
	let numImages: Int
}
