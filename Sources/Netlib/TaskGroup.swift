//******************************************************************************
//  Created by Edward Connell on 6/24/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
import Foundation
import Dispatch

//==============================================================================
// Task
public protocol ModelTask : ModelObject {
	// properties
	var completedItems: Int { get }
	var elapsedTime: TimeInterval { get }
	var estimatedRemainingTime: TimeInterval { get }
	var isCancelled: Bool { get set }
	var maxReportFrequency: TimeInterval { get }
	var totalItems: Int { get }

	// functions
	func cancel()
}

public typealias WorkGroup = (taskGroup: TaskGroup, dispatchGroup: DispatchGroup?)
public typealias TaskArrayHandler = ([UInt8]) -> Void
public typealias TaskUriHandler = (URL) -> Void

//------------------------------------------------------------------------------
// ModelTaskBase
open class ModelTaskBase : ModelObjectBase {
	//----------------------------------------------------------------------------
	// properties
	public var completedItems = 0                       { didSet{onSet("completedItems")} }
	public var elapsedTime: TimeInterval = 0            { didSet{onSet("elapsedTime")} }
	public var estimatedRemainingTime: TimeInterval = 0 { didSet{onSet("estimatedRemainingTime")} }
	public var maxReportFrequency: TimeInterval = 1     { didSet{onSet("maxReportFrequency")} }
	public var totalItems = 0                           { didSet{onSet("totalItems")} }
	public var isCancelled = false

	//----------------------------------------------------------------------------
	// addAccessors
	public override func addAccessors() {
		super.addAccessors()
		addAccessor(name: "completedItems",
		            get: { [unowned self] in self.completedItems },
		            set: { [unowned self] in self.completedItems = $0 })
		addAccessor(name: "elapsedTime",
		            get: { [unowned self] in self.elapsedTime },
		            set: { [unowned self] in self.elapsedTime = $0 })
		addAccessor(name: "estimatedRemainingTime",
		            get: { [unowned self] in self.estimatedRemainingTime },
		            set: { [unowned self] in self.estimatedRemainingTime = $0 })
		addAccessor(name: "maxReportFrequency",
		            get: { [unowned self] in self.maxReportFrequency },
		            set: { [unowned self] in self.maxReportFrequency = $0 })
		addAccessor(name: "totalItems",
		            get: { [unowned self] in self.totalItems },
		            set: { [unowned self] in self.totalItems = $0 })
	}

	//-----------------------------------
	// cancel
	public func cancel() { isCancelled = true }

	//-----------------------------------
	// download data async
	public static func download(uri: Uri, group: WorkGroup,
	                            handler: @escaping TaskArrayHandler) throws {
		// check if item is already cached
		if let url = try uri.getCacheFileURL(makeFolders: false),
			FileSystem.fileExists(path: url.path).exists {

			do {
				if uri.willLog(level: .diagnostic) {
					uri.diagnostic("Loading cached data: \(url.path)",
						categories: [.download, .setup])
				}

				var data = try [UInt8](contentsOf: url)
				if url.path.contains(".gz") {
					uri.diagnostic("unzipping: \(url.path)",
						categories: [.download, .setup], indent: 1)
					data = try unzip(data: data)
				}
				handler(data)

			} catch {
				uri.writeLog("Unable to load cache file: \(error) - \(uri.string)")
				throw error
			}
		} else {
			let downloadTask = DownloadTask()
			group.taskGroup.queue.sync() {
				group.taskGroup.dependents.append(downloadTask)
			}
			try downloadTask.download(uri: uri, group: group.dispatchGroup,
			                          handler: handler)
		}
	}

	//-----------------------------------
	// download file
	public static func downloadFile(uri: Uri, taskGroup: TaskGroup,
	                                handler: @escaping TaskUriHandler) throws {
		let group = WorkGroup(taskGroup, DispatchGroup())
		try downloadFile(uri: uri, group: group, handler: handler)
		group.dispatchGroup!.wait()
	}

	//-----------------------------------
	// download file async
	public static func downloadFile(uri: Uri, group: WorkGroup,
	                                handler: @escaping TaskUriHandler) throws {
		// check if item is already cached
		if let url = try uri.getCacheFileURL(makeFolders: false),
			 FileSystem.fileExists(path: url.path).exists {
			
			if uri.willLog(level: .diagnostic) {
				uri.writeLog("Loading cached item: \(url.path)",
					level: .diagnostic)
			}
			handler(url)
			
		} else {
			let downloadTask = DownloadTask()
			group.taskGroup.queue.sync() {
				group.taskGroup.dependents.append(downloadTask)
			}
			try downloadTask.download(uri: uri, group: group.dispatchGroup,
			                          handler: handler)
		}
	}
}

//==============================================================================
// TaskGroup
public protocol ModelTaskGroup : ModelTask {
	var dependents: [ModelTask] { get set }
	var groupName: String { get set }
}

open class ModelTaskGroupBase : ModelTaskBase {
	
	//----------------------------------------------------------------------------
	// properties
	
	// ModelTaskGroup
	public var dependents = [ModelTask]()          { didSet{onSet("dependents")} }
	public var groupName = ""                      { didSet{onSet("groupName")} }
	public let queue = DispatchQueue(label: "TaskGroup.queue")

	//----------------------------------------------------------------------------
	// addAccessors
	public override func addAccessors() {
		super.addAccessors()
		addAccessor(name: "dependents",
			get: { [unowned self] in self.dependents },
			set: { [unowned self] in self.dependents = $0 })
		addAccessor(name: "groupName" ,
			get: { [unowned self] in self.groupName },
			set: { [unowned self] in self.groupName = $0 })
	}

	// addDependent
	public func addDependent<T: ModelTask>(task: T) -> T {
		queue.sync() {
			dependents.append(task)
		}
		return task
	}
}

//==============================================================================
// TaskGroup
final public class TaskGroup : ModelTaskGroupBase, ModelTaskGroup {
	public convenience init(groupName: String) {
		self.init()
		self.groupName = groupName
	}
}




