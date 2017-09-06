//******************************************************************************
//  Created by Edward Connell on 2/15/17
//  Copyright © 2016 Connell Research. All rights reserved.
//
import Foundation
import Dispatch

final public class TinyImageNet : DataSourceBase, DataSource, InitHelper {
	// initializers
	public required init() {
		uri = Uri(string: "http://cs231n.stanford.edu/tiny-imagenet-200.zip")
		super.init()
		containerTemplate.codecType = .image
	}

	//----------------------------------------------------------------------------
	// DataItem
	struct DataItem {
		var url: URL
		var label: Int?
		var box: (Int, Int, Int, Int)?
	}

	// DataSet
	public enum DataSet : String, EnumerableType { case training, validation, test }

	//----------------------------------------------------------------------------
	// properties
	public var containerTemplate = DataContainer() { didSet{onSet("containerTemplate")} }
	public var dataSet: DataSet?                   { didSet{onSet("dataSet")} }
	public var maxCount = Int.max                  { didSet{onSet("maxCount")} }
	public var uri: Uri					                   { didSet{onSet("uri")} }

	// count
	public var count: Int {
		return min(dataItems.count, maxCount)
	}

	// local data
	private var dataItems = [DataItem]()
	private var wnidIdIndex = [String : Int]()

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
		addAccessor(name: "uri",
		            get: { [unowned self] in self.uri },
		            set: { [unowned self] in self.uri = $0 })
	}

	//----------------------------------------------------------------------------
	// setupData
	public func setupData(taskGroup: TaskGroup) throws {
		// validate
		guard let dataSet = dataSet else {
			writeLog("\(namespaceName): dataSet must be specified")
			throw ModelError.setupFailed
		}

		if dataSet == .training || dataSet == .validation {
			containerTemplate.label = Label()
			containerTemplate.label!.codec = DataCodec(compression: 0)
		}

		//---------------------------------
		// check if data exists, if not download and extract
		let cacheDirPath = NSString(string: uri.cacheDir ?? "/tmp").expandingTildeInPath
		let extracted    = URL(fileURLWithPath: cacheDirPath, isDirectory: true)
		let extractedURL = extracted.appendingPathComponent("tiny-imagenet-200")

		if !FileSystem.fileExists(path: extractedURL.path).exists {
			// create work group for dependents and download the data
			var zipUrl: URL!
			try DownloadTask.downloadFile(uri: self.uri, taskGroup: taskGroup) { zipUrl = $0 }

			// extract to cache
			writeLog("extracting archive to: \(extractedURL.path)", level: .status)
			try extract(zip: zipUrl, to: extracted, totalItems: &taskGroup.totalItems) {
				taskGroup.completedItems = $0 + 1
			}
		}

		// load the wnid <--> label indexes
		try getLabelIndexes(dir: extractedURL)

		//---------------------------------
		// build random access data item list
		switch dataSet {
		case .training:
			let url = extractedURL.appendingPathComponent("train")
			dataItems = try getTrainingSet(dir: url)

		case .validation:
			let url = extractedURL.appendingPathComponent("val")
			try getValidationItems(dir: url, items: &dataItems)

		case .test:
			let url = extractedURL.appendingPathComponent("test")
			try getTestItems(dir: url, items: &dataItems)
		}
	}

	//----------------------------------------------------------------------------
	// getLabelIndexes
	//  If not cached already, this will recurse through the directory
	// structure building wnid indexes then persist the info
	private func getLabelIndexes(dir: URL) throws {
		// build legend
		let wnidData = try Data(contentsOf: dir.appendingPathComponent("wnids.txt"))
		legend = String(data: wnidData, encoding: .utf8)!
			.components(separatedBy: "\n").filter { !$0.isEmpty }
			.map { ["wnid" : $0.trim()] }

		// build wnidId index
		wnidIdIndex.removeAll()
		for i in 0..<legend!.count {
			wnidIdIndex[legend![i]["wnid"]!] = i
		}

		// add words to legend
		let wordData = try Data(contentsOf: dir.appendingPathComponent("words.txt"))
		let words = String(data: wordData, encoding: .utf8)!
			.components(separatedBy: "\n").filter { !$0.isEmpty }
		for str in words {
			let comp = str.components(separatedBy: "\t")
			if let id = wnidIdIndex[comp[0]] {
				legend![id]["words"] = comp[1]
			}
		}
	}

	//----------------------------------------------------------------------------
	// getTrainingSet
	private func getTrainingSet(dir: URL) throws -> [DataItem] {
		// Get synsets
		let synsets = try FileManager.default.contentsOfDirectory(
			at: dir, includingPropertiesForKeys: nil, options: [])

		var dataItems = [DataItem]()
		for synsetURL in synsets {
			try dataItems.append(contentsOf: getTrainingItems(dir: synsetURL))
		}
		return dataItems
	}

	//----------------------------------------------------------------------------
	// getTrainingItems
	private func getTrainingItems(dir: URL) throws -> [DataItem] {
		// build box annotations dictionary
		let wnid = dir.lastPathComponent
		let boxesURL = dir.appendingPathComponent("\(wnid)_boxes.txt")
		let boxesData = try Data(contentsOf: boxesURL)
		let boxStrings = String(data: boxesData, encoding: .utf8)!
			.components(separatedBy: "\n").filter { !$0.isEmpty }.map { $0.trim() }
		var boxes = [String : (Int, Int, Int, Int)]()

		for str in boxStrings {
			let comp = str.components(separatedBy: "\t")
			let row  = Int(comp[1])!
			let col  = Int(comp[2])!
			let rows = Int(comp[3])! - row + 1
			let cols = Int(comp[4])! - col + 1
			boxes[comp[0]] = (row, col, rows, cols)
		}

		// enumerate image files
		let imagesURL = dir.appendingPathComponent("images")
		let images = try FileManager.default.contentsOfDirectory(
			at: imagesURL, includingPropertiesForKeys: nil, options: [])

		return images.map { imageURL in
			DataItem(url: imageURL, label: wnidIdIndex[wnid],
				       box: boxes[imageURL.lastPathComponent]!)
		}
	}

	//----------------------------------------------------------------------------
	// getValidationItems
	private func getValidationItems(dir: URL, items: inout [DataItem]) throws {
		// build box annotations dictionary
		let annotationsURL = dir.appendingPathComponent("val_annotations.txt")
		let annotationData = try Data(contentsOf: annotationsURL)
		let annotationStrings = String(data: annotationData, encoding: .utf8)!
			.components(separatedBy: "\n").filter { !$0.isEmpty }.map { $0.trim() }
		var annotations = [String : (String, (Int, Int, Int, Int))]()
		for str in annotationStrings {
			let comp = str.components(separatedBy: "\t")
			let wnid = comp[1]
			let row  = Int(comp[2])!
			let col  = Int(comp[3])!
			let rows = Int(comp[4])! - row + 1
			let cols = Int(comp[5])! - col + 1
			annotations[comp[0]] = (wnid, (row, col, rows, cols))
		}

		// enumerate image files
		let imagesURL = dir.appendingPathComponent("images")
		let images = try FileManager.default.contentsOfDirectory(
			at: imagesURL, includingPropertiesForKeys: nil, options: [])

		for imageURL in images {
			let anno = annotations[imageURL.lastPathComponent]!
			let dataItem = DataItem(url: imageURL, label: wnidIdIndex[anno.0], box: anno.1)
			items.append(dataItem)
		}
	}

	//----------------------------------------------------------------------------
	// getTestItems
	private func getTestItems(dir: URL, items: inout [DataItem]) throws {
		// enumerate image files
		let imagesURL = dir.appendingPathComponent("images")
		let images = try FileManager.default.contentsOfDirectory(
			at: imagesURL, includingPropertiesForKeys: nil, options: [])

		for imageURL in images {
			let dataItem = DataItem(url: imageURL, label: nil, box: nil)
			items.append(dataItem)
		}
	}

	//----------------------------------------------------------------------------
	// getItem
	public func getItem(at index: Int, taskGroup: TaskGroup) throws -> ModelLabeledContainer {
		assert(index < count, "Index out of range")
		let item = dataItems[index]

		return try containerTemplate.copyTemplate(index: index, taskGroup: taskGroup) {
			$0.uri = Uri(url: item.url)
			
			if let label = $0.label {
				label.sourceIndex = index
				label.value = [Double(item.label!)]

				if let box = item.box {
					label.rangeOffset = [1, 1, box.0, box.1]
					label.rangeExtent = [1, 1, box.2, box.3]
				}
			} else if let itemLabel = item.label {
				$0.labelValue = [Double(itemLabel)]
			}
		}
	}
}

