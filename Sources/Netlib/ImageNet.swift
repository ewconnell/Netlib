//******************************************************************************
// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
import Foundation
import Dispatch

final public class ImageNet : DataSourceBase, DataSource, InitHelper {
	// initializers
	public required init() {
		// set some default values
		inetStructureUri = Uri(
			string: "http://www.image-net.org/api/xml/structure_released.xml")
		selectionMethod = .sequential

		super.init()
		containerTemplate.codecType = .image
	}

	//----------------------------------------------------------------------------
	// SelectionMethod
	public enum SelectionMethod : String, EnumerableType { case sequential, random }

	// DataSet
	public enum DataSet : String, EnumerableType { case training, validation, test }

	//----------------------------------------------------------------------------
	// properties
	public var containerTemplate = DataContainer() { didSet{onSet("containerTemplate")} }
	public var inetStructureUri: Uri               { didSet{onSet("structUri")} }
	public var maxCount = Int.max                  { didSet{onSet("maxCount")} }
	public var maxItemsPerCategory = 1000          { didSet{onSet("maxItemsPerCategory")} }
	public var selectedCategories = [String]()     { didSet{onSet("selectedCategories")} }
	public var selectionMethod: SelectionMethod    { didSet{onSet("selectionMethod")} }
	public var validationPercent = 0.30            { didSet{onSet("validationPercent")} }
	public var dataSet: DataSet?

	// count
	public var count: Int {
		return 0
//		return min(imagesHeader?.numImages ?? 0, maxCount)
	}

	// local data
	private var urlList = [URL]()
	private var selectionTree: ImgNetNode?
	private let selectionTreeUri = Uri()
	private var downloadWorkGroup: WorkGroup!

	//----------------------------------------------------------------------------
	// addAccessors
	public override func addAccessors() {
		super.addAccessors()
		addAccessor(name: "containerTemplate",
		            get: { [unowned self] in self.containerTemplate },
		            set: { [unowned self] in self.containerTemplate = $0 })
		addAccessor(name: "inetStructureUri",
		            get: { [unowned self] in self.inetStructureUri },
		            set: { [unowned self] in self.inetStructureUri = $0 })
		addAccessor(name: "maxCount",
		            get: { [unowned self] in self.maxCount },
		            set: { [unowned self] in self.maxCount = $0 })
		addAccessor(name: "maxItemsPerCategory",
		            get: { [unowned self] in self.maxItemsPerCategory },
		            set: { [unowned self] in self.maxItemsPerCategory = $0 })
		addAccessor(name: "selectedCategories",
		            get: { [unowned self] in self.selectedCategories },
		            set: { [unowned self] in self.selectedCategories = $0 })
		addAccessor(name: "selectionMethod",
		            get: { [unowned self] in self.selectionMethod },
		            set: { [unowned self] in self.selectionMethod = $0 })
		addAccessor(name: "validationPercent",
		            get: { [unowned self] in self.validationPercent },
		            set: { [unowned self] in self.validationPercent = $0 })
	}

	//----------------------------------------------------------------------------
	// setup
	public override func setup(taskGroup: TaskGroup) throws {
		try super.setup(taskGroup: taskGroup)
		selectionTreeUri.cacheDir = inetStructureUri.cacheDir
		selectionTreeUri.cacheFile = "inetStructure.json"
		downloadWorkGroup = WorkGroup(taskGroup, DispatchGroup())
	}

	//----------------------------------------------------------------------------
	// setupData
	//
	public func setupData(taskGroup: TaskGroup) throws {
		try buildURLList()

	}

	//----------------------------------------------------------------------------
	// getSelectionTree
	//  this will download the inet structure xml if needed and convert
	// it to a cached json version. Then build a tree with the wnid values
	// and words.
	//
	private func getSelectionTree(completion handler: (ImgNetNode) -> Void) throws {
		guard selectionTree == nil else { handler(selectionTree!); return }

		do {
			// cached inet structure in json form
			let jsonURL = try selectionTreeUri.getCacheFileURL(makeFolders: true)!

			if FileManager.default.fileExists(atPath: jsonURL.path) {
				let jsonData = try Data(contentsOf: jsonURL)
				selectionTree = ImgNetNode()
				try selectionTree!.updateAny(
					with: JSONSerialization.jsonObject(with: jsonData) as? AnySet)

			} else {
				// create work group for dependents and download the data
				var structureXmlData = [UInt8]()
				try DownloadTask.download(uri: inetStructureUri, group: downloadWorkGroup) {
					structureXmlData = $0
				}
				downloadWorkGroup.dispatchGroup!.wait()

				// build category tree
				let doc = try XmlDocument(log: currentLog, data: structureXmlData)
				if let root = doc.rootElement {
					guard root.children.count == 2 && root.children[1].name == "synset" else {
						writeLog("image net structure xml has unexpected root layout")
						throw ModelError.setupFailed
					}
					selectionTree = addNode(element: root.children[1])

				} else {
					writeLog("image net structure xml has no root")
					throw ModelError.setupFailed
				}

				// save as json
				let selected = selectionTree!.selectAny(include: .types) ?? AnySet()
				let data = try! JSONSerialization.data(withJSONObject: selected, options: [])
				let json = String(data: data, encoding: .utf8)!
				try json.write(to: jsonURL, atomically: true, encoding: .utf8)
			}

			handler(selectionTree!)

		} catch {
			writeLog("Failed to build image net structure")
			writeLog(String(describing: error))
			throw ModelError.setupFailed
		}
	}

	//------------------------------------
	// addNode
	private func addNode(element: XmlElement) -> ImgNetNode
	{
		let node = ImgNetNode()
		node.model = model
		for attr in element.attributes {
			switch attr.name {
			case "words": node.words = attr.stringValue
			case "wnid": node.wnid = attr.stringValue
			default: break
			}
		}

		for child in element.children {
			node.subsets.append(addNode(element: child))
		}

		return node
	}

	//----------------------------------------------------------------------------
	// buildURLList
	//  this walks the selection tree and accumulates the list of URLs
	private func buildURLList() throws
	{
		let queryStr = "http://www.image-net.org/api/text/imagenet.synset.geturls?wnid="

		urlList.removeAll()
		let queue = DispatchQueue(label: "ImageNet.queue")

		for wnid in selectedCategories {
			let query = Uri(string: queryStr + wnid)
			try DownloadTask.download(uri: query, group: downloadWorkGroup) {
				// convert the response into a URL array
				let str = String(bytes: $0, encoding: .utf8)!
				let urls: [URL] = str.components(separatedBy: "\r").map { URL(string: $0 )!}

				queue.sync {
					self.urlList.append(contentsOf: urls)
				}
			}
		}
		downloadWorkGroup.dispatchGroup!.wait()
	}


	//----------------------------------------------------------------------------
	// getItem
	public func getItem(at index: Int, taskGroup: TaskGroup) throws -> ModelLabeledContainer {
		fatalError()
	}
}

//------------------------------------------------------------------------------
// ImageInfo
public enum SelectionState : String, EnumerableType {
	case selected, partial, notSelected
}

public struct ImageInfo {
	var state: SelectionState
	var image: Uri
}

//------------------------------------------------------------------------------
// ImgNetNode
public final class ImgNetNode : ModelObjectBase
{
	//----------------------------------------------------------------------------
	// properties
	public var images  = [ImageInfo]()
	public var state   = SelectionState.notSelected { didSet{onSet("state")} }
	public var subsets = [ImgNetNode]()             { didSet{onSet("subsets")}}
	public var words   = ""                         { didSet{onSet("words")} }
	public var wnid    = ""                         { didSet{onSet("wnid")} }

	//----------------------------------------------------------------------------
	// addAccessors
	public override func addAccessors() {
		super.addAccessors()
		addAccessor(name: "state"  , get: { [unowned self] in self.state }, set: { [unowned self] in self.state = $0 })
		addAccessor(name: "subsets", get: { [unowned self] in self.subsets }, set: { [unowned self] in self.subsets = $0 })
		addAccessor(name: "wnid"   , get: { [unowned self] in self.wnid }, set: { [unowned self] in self.wnid = $0 })
		addAccessor(name: "words"  , get: { [unowned self] in self.words }, set: { [unowned self] in self.words = $0 })
	}

	//----------------------------------------------------------------------------
	// getImageURLs
	public func getImageInfo(completion handler: (ImgNetNode) -> Void) throws {
		guard images.count == 0 else { handler(self); return }


	}
}
