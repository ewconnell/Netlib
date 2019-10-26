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

final public class FileList : DataSourceBase, DataSource {
	//----------------------------------------------------------------------------
	// properties
	public var containerTemplates = [String : DataContainer]() { didSet{onSet("containerTemplates")} }
	public var directory: String?                  { didSet{onSet("directory")} }
	public var fileTypes: [String : CodecType]?    { didSet{onSet("fileTypes")} }
	public var filter = [String]()                 { didSet{onSet("filter")} }
	public var includeSubDirs = true               { didSet{onSet("includeSubDirs")}}
	public var items = [DataContainer]()           { didSet{onSet("items")} }
	public var maxCount = Int.max                  { didSet{onSet("maxCount")} }

	// count
	public var count: Int {
		return min(items.count, maxCount)
	}

	// local data
	private let queue = DispatchQueue(label: "FileList.queue")
	private var filterSet: Set<String>!
	private var taskGroup: TaskGroup!
	private var extFileTypes: [String : CodecType] =
	[
		"jpg"  : .image, "jpeg" : .image, "png" : .image,
		"pbm"  : .image, "pgm"  : .image, "ppm" : .image,
		"mp4a" : .audio, "wav"  : .audio
	]

	//----------------------------------------------------------------------------
	// addAccessors
	public override func addAccessors() {
		super.addAccessors()
		addAccessor(name: "containerTemplates",
		            get: { [unowned self] in self.containerTemplates },
		            set: { [unowned self] in self.containerTemplates = $0 })
		addAccessor(name: "directory",
		            get: { [unowned self] in self.directory },
		            set: { [unowned self] in self.directory = $0 })
		addAccessor(name: "fileTypes",
		            get: { [unowned self] in self.fileTypes },
		            set: { [unowned self] in self.fileTypes = $0 })
		addAccessor(name: "filter",
		            get: { [unowned self] in self.filter },
		            set: { [unowned self] in self.filter = $0 })
		addAccessor(name: "includeSubDirs",
		            get: { [unowned self] in self.includeSubDirs },
		            set: { [unowned self] in self.includeSubDirs = $0 })
		addAccessor(name: "items",
		            get: { [unowned self] in self.items },
		            set: { [unowned self] in self.items = $0 })
		addAccessor(name: "maxCount",
		            get: { [unowned self] in self.maxCount },
		            set: { [unowned self] in self.maxCount = $0 })
	}

	//----------------------------------------------------------------------------
	// setupData
	public func setupData(taskGroup: TaskGroup) throws {
		// save for use in the subscript method
		self.taskGroup = taskGroup

		// merge user extensions
		if let fileTypes = self.fileTypes {
			for fileType in fileTypes {	extFileTypes[fileType.key] = fileType.value	}
		}
		
		// if items was not explicitly set, build DataContainer list from directory
		if items.count == 0 {
			if let dir = directory {
				filterSet = Set<String>(filter)
				let dirPath = NSString(string: dir).expandingTildeInPath
				let dirURL = URL(fileURLWithPath: dirPath, isDirectory: true)
				
				// verify folder
				if !FileSystem.fileExists(path: dirURL.path).exists {
					writeLog("\(dirPath) is not found")
					throw ModelError.setupFailed
				}
				
				// enumerate
				addItems(fileManager: FileManager.default, dir: dirURL)
			} else if let storage = try findStorageLocation() {
				// enumerate
				filterSet = Set<String>(filter)
				addItems(fileManager: FileManager.default, dir: storage)
			} else {
				writeLog("\(namespaceName) - Data directory must be specified")
				throw ModelError.setupFailed
			}
		}
	}

	//----------------------------------------------------------------------------
	// addFolderItems
	private func addItems(fileManager: FileManager, dir: URL) {
		// get file list
		#if os(Linux)
			let urls = try! fileManager.contentsOfDirectory(
				at: dir, includingPropertiesForKeys: nil)
		#else
			let urls = try! fileManager.contentsOfDirectory(
				at: dir, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles])
		#endif
		
		for url in urls {
			if includeSubDirs {
				if FileSystem.fileExists(path: url.path).isDir {
					addItems(fileManager: fileManager, dir: url)
				} else {
					addItem(url: url)
				}
			} else {
				addItem(url: url)
			}
		}
	}

	//----------------------------------------------------------------------------
	// addItem
	private func addItem(url: URL) {
		let ext = url.pathExtension
		if filterSet.contains(ext) {
			guard extFileTypes[ext] != nil else {
				writeLog("Unknown file type '\(url.pathExtension)' ignored")
				return
			}

			// create a template of the correct type if needed
			let codecType = extFileTypes[url.pathExtension]!
			if containerTemplates[url.pathExtension] == nil {
				let temp = DataContainer()
				temp.codecType = codecType
				containerTemplates[url.pathExtension] = temp
			}

			let container = containerTemplates[url.pathExtension]!.copy() 
			container.uri = Uri(url: url)
			items.append(container)
		}
	}


	//----------------------------------------------------------------------------
	// getItem
	public func getItem(at index: Int, taskGroup: TaskGroup) throws -> ModelLabeledContainer {
		assert(index < count, "Index out of range")
		let container = items[index]
		container.sourceIndex = index
		try container.setup(taskGroup: taskGroup)
		return container
	}
}
