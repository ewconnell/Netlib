//******************************************************************************
//  Created by Edward Connell on 4/25/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
import Foundation
import Dispatch
import Lmdb

//==============================================================================
// Database
final public class Database : ModelElementBase, MetricsElement, InitHelper {
	// initializers
	public required init() {
		super.init()
		createTaskGroup = true
		outputs["data"] = Connector()
		outputs["labels"] = Connector()
		let prop = properties["outputs"]!
		prop.isGenerated = true
		prop.isCopyable = false

		// TODO: remove
		// keep these defaulted now until cudnn format problems are sorted out
		dataType = .real32F
		dataLayout = .nchw
	}

	//----------------------------------------------------------------------------
	// enums
	public enum Rebuild : String, EnumerableType { case always, create, never	}
	public enum Uniformity : String, EnumerableType {
		case any, numberOfChannels, extent
	}
	
	//----------------------------------------------------------------------------
	// SelectionInfo
	struct SelectionInfo {
		let firstItem: DataContainer
		let dataKeys: [DataKey]
	}
	
	// OutputState
	class OutputState {
		var totalFetchedCount = 0

		// cache data for cached forward mode
		var cachedData: DataView?
		var cachedLabels: DataView?
		var cachedAnnotations: [Label?]?
		var nextCacheIndex = 0

		// prefetch data for streaming forward mode
		var prefetchData: DataView?
		var prefetchLabels: DataView?
		var prefetchAnnotations: [Label?]?
		var prefetchError: Error?
	}
	
	//----------------------------------------------------------------------------
	// properties
	public var connection: String?                   { didSet{onSet("connection")} }
	public var dataLayout: DataLayout?               { didSet{onSet("dataLayout")} }
	public var dataType: DataType?                   { didSet{onSet("dataType")} }
	public var labelDataType: DataType?              { didSet{onSet("labelDataType")} }
	public var provider: DbProvider =	LmdbProvider() { didSet{onSet("provider")} }
	public var rebuild = Rebuild.create              { didSet{onSet("rebuild")} }
	public var requireUniform = Uniformity.extent    { didSet{onSet("requireUniform")} }
	public var shuffle = true                        { didSet{onSet("shuffle")} }
	public var source: DataSource?                   { didSet{onSet("source")} }
	public var streamOutput = false                  { didSet{onSet("streamOutput")} }
	public var transform: Transform?                 { didSet{onSet("transform")} }
	public var writeBufferSize = 10.MB               { didSet{onSet("writeBufferSize")} }
	public var validateBuild = false                 { didSet{onSet("validateBuild")} }

	// ~/Documents/data
	public var dataDir: String = getDataDirectory() {
		didSet {
			provider.dataDir = dataDir
			onSet("dataDir")
		}
	}

	// for fetching
	private var streamingDbSession: DbSession!
	private var dataTableCount = 0
	private let fetchQueue = DispatchQueue(label: "Database.queue")
	private var nextPrefetchKey: DataKey?
	private var numOutputs: Int { return streams?.count ?? 0 }
	private lazy var outputItems: [OutputState] = {
		var array = [OutputState]()
		for _ in 0..<self.numOutputs { array.append(OutputState()) }
		return array
	}()

	//----------------------------------------------------------------------------
	// addAccessors
	public override func addAccessors() {
		super.addAccessors()
		addAccessor(name: "connection",
		            get: { [unowned self] in self.connection },
		            set: { [unowned self] in self.connection = $0 })
		addAccessor(name: "dataLayout",
			          get: { [unowned self] in self.dataLayout },
			          set: { [unowned self] in self.dataLayout = $0 })
		addAccessor(name: "dataType",
			          get: { [unowned self] in self.dataType },
			          set: { [unowned self] in self.dataType = $0 })
		addAccessor(name: "dataDir",
		            get: { [unowned self] in self.dataDir },
		            set: { [unowned self] in self.dataDir = $0 })
		addAccessor(name: "labelDataType",
			          get: { [unowned self] in self.labelDataType },
			          set: { [unowned self] in self.labelDataType = $0 })
		addAccessor(name: "provider",
		            get: { [unowned self] in self.provider },
		            set: { [unowned self] in self.provider = $0 })
		addAccessor(name: "rebuild",
		            get: { [unowned self] in self.rebuild },
		            set: { [unowned self] in self.rebuild = $0 })
		addAccessor(name: "requireUniform",
		            get: { [unowned self] in self.requireUniform },
		            set: { [unowned self] in self.requireUniform = $0 })
		addAccessor(name: "shuffle",
		            get: { [unowned self] in self.shuffle },
		            set: { [unowned self] in self.shuffle = $0 })
		addAccessor(name: "source",
		            get: { [unowned self] in self.source },
		            set: { [unowned self] in self.source = $0 })
		addAccessor(name: "streamOutput",
		            get: { [unowned self] in self.streamOutput },
		            set: { [unowned self] in self.streamOutput = $0 })
		addAccessor(name: "transform",
		            get: { [unowned self] in self.transform },
		            set: { [unowned self] in self.transform = $0 })
		addAccessor(name: "writeBufferSize",
		            get: { [unowned self] in self.writeBufferSize },
		            set: { [unowned self] in self.writeBufferSize = $0 })
		addAccessor(name: "validateBuild",
		            get: { [unowned self] in self.validateBuild },
		            set: { [unowned self] in self.validateBuild = $0 })
	}

	//----------------------------------------------------------------------------
	// resetMetrics
	public func resetMetrics() {
		for item in outputItems { item.totalFetchedCount = 0 }
	}

	//----------------------------------------------------------------------------
	// copy
	public override func copy(from other: Properties) {
		super.copy(from: other)
		let other = other as! Database
		if !other.streamOutput {
			outputItems = other.outputItems.map {
				let state = OutputState()
				state.cachedData = $0.cachedData
				state.cachedLabels = $0.cachedLabels
				state.cachedAnnotations = $0.cachedAnnotations
				return state
			}
		}
	}

	//----------------------------------------------------------------------------
	// setup
	public override func setup(taskGroup: TaskGroup) throws {
		// setup base and dependents
		try super.setup(taskGroup: taskGroup)

		// explicitly set connection and dataDir on provider
		// because it may not be done during template application
		provider.connection = connection ?? namespaceName
		provider.dataDir = dataDir

		// check uniformity
		if requireUniform != .extent && !streamOutput {
			streamOutput = true
			if willLog(level: .warning) {
				let message = "\(namespaceName) databases with non-uniform item extents require" +
				" the output to be streamed. Property 'streamOutput' is set to 'true'"
				writeLog(message, level: .warning)
			}
		}

		// since this is a leaf node, create some streams to work with
		streams = try model!.compute.requestStreams(
			label: "\(namespaceName) dataStream",
			serviceName: computeServiceName, deviceIds: deviceIds)

		// create a connector item for each output stream
		let dataConnector = outputs["data"]!
		let labelsConnector = outputs["labels"]!

		// reinitialize in case of multiple setup
		dataConnector.items.removeAll()
		labelsConnector.items.removeAll()

		for stream in streams! {
			dataConnector.items.append(ConnectorItem(using: stream))
			labelsConnector.items.append(ConnectorItem(using: stream))
		}

		// reset fetch state
		for item in outputItems {
			item.totalFetchedCount = 0
			item.nextCacheIndex = 0
		}

		// make sure the database is built
		if rebuild == .always || (rebuild == .create && !provider.databaseExists) {
			try rebuildDatabase(taskGroup: taskGroup)
		}
	}

	//----------------------------------------------------------------------------
	// setupForward
	public override func setupForward(mode: EvaluationMode, selection: Selection) throws {
		try super.setupForward(mode: mode, selection: selection)

		if streamOutput || selection.usesSpecificKeys {
			// get the number of items in the data table
			streamingDbSession = try provider.open(mode: .readOnly)
			dataTableCount = try streamingDbSession.dataTable.count()

			if !selection.wrapAround && dataTableCount > selection.count &&
				   dataTableCount % selection.count != 0
			{
				writeLog("\(namespaceName) selection count(\(selection.count)) is not a " +
					"multiple of item count(\(dataTableCount)). " +
					"Testing accuracy will not be exactly correct.", level: .warning)
			}

		} else {
			try setupCachedForward(selection: selection)
		}
	}

	//----------------------------------------------------------------------------
	// setupCachedForward
	//	The entire data set is loaded into memory and distributed
	// equally among the selected devices/streams
	private func setupCachedForward(selection: Selection) throws {
		assert(requireUniform == .extent)

		// open
		let session = try provider.open(mode: .readOnly)
		dataTableCount = try session.dataTable.count()

		// divide the selection between the streams
		var totalItems = dataTableCount
		let multiple = selection.count * numOutputs
		if selection.wrapAround && totalItems % multiple != 0 {
			totalItems = ((totalItems / multiple) + 1) * multiple
		}

		var itemsPerStream: Int
		var itemsPerLastStream: Int
		if totalItems % numOutputs == 0 {
			itemsPerStream = totalItems / numOutputs
			itemsPerLastStream = itemsPerStream
		} else {
			itemsPerStream = totalItems / numOutputs + 1
			itemsPerLastStream = itemsPerStream - totalItems % numOutputs
		}

		// if this is templated, then find other
		let templateDb = try findTemplate() as? Database

		// internal cache/prefetch state is maintained for each
		// item on the data/labels output connectors
		let dataConnector = outputs["data"]!

		for (i, state) in outputItems.enumerated() {
			// if this is a templated database object, check if the data has already
			// been computed by the template source object. If so, copy it
			if let other = templateDb?.outputItems[i] {
				if other.cachedData != nil {
					if willLog(level: .diagnostic) {
						diagnostic("\(namespaceName) using cached output from template " +
							"\(templateDb!.namespaceName) items: \(other.cachedData!.shape.extent[0])",
							categories: [.setup, .setupForward])
					}
					state.cachedData = other.cachedData
					state.cachedLabels = other.cachedLabels
					state.cachedAnnotations = other.cachedAnnotations
				}
			}

			// if the cache is still empty, then fetch the data
			if state.cachedData == nil {
				if willLog(level: .diagnostic) {
					diagnostic("\(namespaceName) computing cached output",
						         categories: [.setup, .setupForward])
				}

				// use the device stream associated with this item
				let stream = dataConnector.items[i].stream

				// the last stream may have less than a full count
				let selectionCount = (i == outputItems.count - 1) ?
					itemsPerLastStream : itemsPerStream

				// create the selection
				// the selection might be more than the actual available items to
				// be a multiple of batch size, so it must be allowed to wrap around
				let cachedSelection = Selection(count: selectionCount, wrapAround: true)

				// retrieve the data
				state.totalFetchedCount = try fetch(
					selection: cachedSelection,
					session: session,
					outData: &state.cachedData,
					outLabels: &state.cachedLabels,
					annotations: &state.cachedAnnotations,
					using: stream)

				// the cache data is loaded and transferred to the device address
				// space only once, so release the uma buffer after that happens
				state.cachedData!.dataArray.autoReleaseUmaBuffer = true
				state.cachedLabels?.dataArray.autoReleaseUmaBuffer = true

				// async push to the device while waiting to get the other outputs
				if !stream.device.usesUnifiedAddressing {
					_ = try state.cachedData!.ro(using: stream)
					_ = try state.cachedLabels?.ro(using: stream)
				}
			}
		}
	}

	//----------------------------------------------------------------------------
	// forward
	public override func forward(mode: EvaluationMode,
	                             selection: Selection) throws -> FwdProgress {
		// if the data is already on the outputs then return
		guard selection != currentFwdSelection else { return currentFwdProgress }

		// setup the element's forward pass if needed
		if doSetupForward {	try setupForward(mode: mode, selection: selection) }

		if selection.usesSpecificKeys {
			return try keysForward(selection: selection)

		} else if streamOutput {
			return try streamForward(selection: selection)

		} else {
			return try cachedForward(selection: selection)
		}
	}

	//----------------------------------------------------------------------------
	// cachedForward
	//	All of the data has been loaded, so this returns the selected view
	private func cachedForward(selection: Selection) throws -> FwdProgress {
		if willLog(level: .diagnostic) {
			diagnostic("\(namespaceName).cachedForward", categories: .evaluate)
		}

		// make sure there is something to work with
		guard dataTableCount > 0 else {
			writeLog("cachedForward failed - the database is empty")
			throw ModelError.evaluationFailed
		}

		// get to avoid multiple dictionary lookups
		let dataConnector = outputs["data"]!
		let labelsConnector = outputs["labels"]!

		for i in 0..<numOutputs {
			let item = outputItems[i]
			let dataShape = item.cachedData!.shape

			// determine the count
			var dataCount: Int
			// training selections want wrap around
			if selection.wrapAround {
				dataCount = dataShape.extent[0]
			} else {
				// testing selections do not want wrap around,
				// so clamp to actual number of database items
				dataCount = min(dataShape.extent[0], dataTableCount / numOutputs)
			}
			assert(dataCount > 0)

			// clamp selection count
			let selectionCount = min(selection.count, dataCount)

			// if the end will be passed then wrap around
			if item.nextCacheIndex + selectionCount >= dataCount {
				item.nextCacheIndex = 0
			}

			// return a view of the cached data at the specified index position
			dataConnector.items[i].data = item.cachedData!
				.viewItems(offset: item.nextCacheIndex, count: selectionCount)

			// DEBUG
//			print(dataConnector.items[i].data.format(
//				columnWidth: 2, precision: 0, maxItems: 5, highlightThreshold: 0))

			// optional labels
			if let labels = item.cachedLabels {
				labelsConnector.items[i].data = labels
					.viewItems(offset: item.nextCacheIndex, count: selectionCount)

				// DEBUG
//				print(labelsConnector.items[i].data.format(maxItems: 5))
			}

			// progress
			item.nextCacheIndex += selectionCount
			item.totalFetchedCount += selectionCount
			currentFwdProgress =
				FwdProgress(epoch: Double(item.totalFetchedCount) / Double(dataCount))
		}

		// this is the new current selection
		currentFwdSelection = selection
		return currentFwdProgress
	}

	//----------------------------------------------------------------------------
	// keysForward
	//	This loads the selection from the database on demand with
	// asynchronous prefetch
	private func keysForward(selection: Selection) throws -> FwdProgress {
		if willLog(level: .diagnostic) {
			diagnostic("\(namespaceName).keysForward", categories: .evaluate)
		}

		// internal cache/prefetch state is maintained for each
		// item on the data/labels output connectors
		let dataConnector = outputs["data"]!
		let labelsConnector = outputs["labels"]!

		for i in 0..<numOutputs {
			// check last async error
			let stream     = dataConnector.items[i].stream
			let state      = outputItems[i]
			let dataItem   = dataConnector.items[i]
			let labelsItem = labelsConnector.items[i]

			state.totalFetchedCount += try fetch(
				selection: selection,
				session: streamingDbSession,
				outData: &state.prefetchData,
				outLabels: &state.prefetchLabels,
				annotations: &state.prefetchAnnotations,
				using: stream)

			// push to device if not UMA
			if !stream.device.usesUnifiedAddressing {
				_ = try state.prefetchData!.ro(using: stream)
				_ = try state.prefetchLabels!.ro(using: stream)
			}

			swap(&dataItem.data, &state.prefetchData!)
			swap(&labelsItem.data, &state.prefetchLabels!)
			currentFwdProgress = FwdProgress(epoch: Double(state.totalFetchedCount) /
				Double(dataTableCount))
		}

		// update current selection and return status
		currentFwdSelection = selection
		return currentFwdProgress
	}

	//----------------------------------------------------------------------------
	// streamForward
	//	This loads the selection from the database on demand with
	// asynchronous prefetch
	private func streamForward(selection: Selection) throws -> FwdProgress {
		if willLog(level: .diagnostic) {
			diagnostic("\(namespaceName).streamForward", categories: .evaluate)
		}

		// internal cache/prefetch state is maintained for each
		// item on the data/labels output connectors
		let dataConnector = outputs["data"]!
		let labelsConnector = outputs["labels"]!

		return try fetchQueue.sync {
			for i in 0..<numOutputs {
				// check last async error
				let stream     = dataConnector.items[i].stream
				let state      = outputItems[i]
				let dataItem   = dataConnector.items[i]
				let labelsItem = labelsConnector.items[i]
				guard state.prefetchError == nil else { throw state.prefetchError! }

				// if there is no prefetch data then synchronously load it
				if state.prefetchData == nil {
					state.totalFetchedCount += try fetch(
						selection: selection,
						session: streamingDbSession,
						outData: &state.prefetchData,
						outLabels: &state.prefetchLabels,
						annotations: &state.prefetchAnnotations,
						using: stream)

					// push to device if not UMA
					if !stream.device.usesUnifiedAddressing {
						_ = try state.prefetchData!.ro(using: stream)
						_ = try state.prefetchLabels!.ro(using: stream)
					}
				}

				// now swap and reload the prefetch asynchronously to be
				// ready for the next forward request
				swap(&dataItem.data, &state.prefetchData!)
				swap(&labelsItem.data, &state.prefetchLabels!)
				currentFwdProgress = FwdProgress(epoch: Double(state.totalFetchedCount) /
					                             Double(dataTableCount))

				// request next batch if specific keys weren't specified
				fetchQueue.async(group: model.asyncTaskGroup) { [unowned self] in
					// create a selection object
					// it will retain the Function during the closure
					do {
						state.totalFetchedCount += try self.fetch(
							selection: selection.next(),
							session: self.streamingDbSession,
							outData: &state.prefetchData,
							outLabels: &state.prefetchLabels,
							annotations: &state.prefetchAnnotations,
							using: stream)

						// push to device if not UMA
						if !stream.device.usesUnifiedAddressing {
							_ = try state.prefetchData!.ro(using: stream)
							_ = try state.prefetchLabels!.ro(using: stream)
						}
					} catch {
						state.prefetchError = error
					}
				}
			}

//			print(dataConnector.items[0].data.format(columnWidth: 3, precision: 1,
//				limitExtent: Extent(10, 1, 28, 28), highlightThreshold: 0))
//
//			print(labelsConnector.items[0].data.format(limitExtent: Extent(10)))
//

			// update current selection and return status
			currentFwdSelection = selection
			return currentFwdProgress
		}
	}

	//----------------------------------------------------------------------------
	// fetch
	//	This retrieves the selection from the database and stores it in the
	// specified data buffers
	private func fetch(selection: Selection,
	                   session: DbSession,
	                   outData: inout DataView?,
	                   outLabels: inout DataView?,
	                   annotations: inout [Label?]?,
	                   using stream: DeviceStream) throws -> Int {
		// validate
		assert(selection.count > 0)
		
		// save selection and query the needed info to construct a result
		let info  = try querySelectionInfo(session: session, selection: selection)
		let keys  = info.dataKeys
		let count = keys.count

		// assure data and label buffers are the correct type and size
		let outDataShape = Shape(items: count, shape: info.firstItem.shape, layout: dataLayout)

		if outData == nil || outData!.shape != outDataShape {
			var output = DataView(shape: outDataShape,
			                      dataType: dataType ?? info.firstItem.dataType)
			output.isShared = true
			output.log = currentLog
			output.name = namePath
			outData = output

			// trigger lazy allocation so it's allocated and freed on the same thread
			_ = try output.roReal8U()
		}
		var dataRef: DataView! = try outData!.reference(using: nil)

		// setup optional labels
		var labelsRef: DataView?

		if let itemLabelShape = info.firstItem.labelShape {
			let labelsShape = Shape(items: count, shape: itemLabelShape)

			if outLabels == nil || outLabels!.shape != labelsShape {
				// the output is either the label value or an index to an annotation
				var outputDataType: DataType
				if info.firstItem.labelValue != nil {
					outputDataType = labelDataType ?? .real32I
				} else {
					outputDataType = .real32I
					annotations = [Label?](repeating: nil, count: count)
				}

				var output = DataView(shape: labelsShape, dataType: outputDataType)
				output.isShared = true
				output.log = currentLog
				output.name = namePath
				outLabels = output

				// trigger lazy allocation so it's allocated and freed on the same thread
				_ = try output.roReal8U()
			}
			labelsRef = try outLabels!.reference(using: nil)
			assert(outData!.items == outLabels!.items)
		}

		//-----------------------------------
		// work function
		let retain = !provider.isMemoryMapped

		func assignItem(itemIndex: Int, txn: DbDataTransaction,
		                dataRef: DataView, labelsRef: DataView?) throws {
			// get the item data and write to output view
			let dbData = try txn.cursor().set(key: keys[itemIndex])!
			var next = BufferUInt8()
			let container = try DataContainer(from: dbData.buffer, next: &next,
			                                  retain: retain)
			annotations?[itemIndex] = container.label

			// optional label sub view
			var labelView = labelsRef?.view(item: itemIndex)

			// decode to data sub view
			var dataView = dataRef.view(item: itemIndex)
			try container.decode(data: &dataView, label: &labelView)
		}

		let txn = try session.dataTable.transaction(mode: .readOnly)

		//--------------------------------------------------------------------------
		// assign each selected item to it's corresponding result position
		if model.concurrencyMode == .concurrent {
			var abort = false
			let abortMutex = PThreadMutex()

			// all iterations are guaranteed complete before leaving this scope
			DispatchQueue.concurrentPerform(iterations: keys.count)
			{ [unowned self, unowned txn, unowned abortMutex] in
				// check for abort condition
				if (abortMutex.fastSync { return abort }) { return }

				do {
					try assignItem(itemIndex: $0, txn: txn,
						             dataRef: dataRef, labelsRef: labelsRef)
				} catch {
					// log the error
					abortMutex.fastSync { abort = true }
					self.writeLog(String(describing: error))
				}
			}
			if abort { throw ModelError.evaluationFailed }

		} else {
			for index in 0..<keys.count {
				try assignItem(itemIndex: index, txn: txn,
					             dataRef: dataRef, labelsRef: labelsRef)
			}
		}

		// release references
		dataRef = nil
		labelsRef = nil
		assert(outData!.isUniqueReference() &&
			    (outLabels == nil || outLabels!.isUniqueReference()))

		return selection.count
	}

	//----------------------------------------------------------------------------
	// querySelectionInfo
	private func querySelectionInfo(session: DbSession,
	                                selection: Selection) throws -> SelectionInfo {
		if let keys = selection.keys {
			return try querySpecificKeysInfo(session: session, keys: keys)
		} else {
			return try queryNextKeysInfo(session: session, selection: selection)
		}
	}

	//----------------------------------------------------------------------------
	// querySpecificKeysInfo
	//  this is to retrieve a specific set of containers by key value
	private func querySpecificKeysInfo(session: DbSession,
	                                   keys: [DataKey]) throws -> SelectionInfo {
		// create a read cursor
		let cursor = try streamingDbSession.dataTable.transaction().cursor()
		guard let firstItem = try cursor.set(key: keys[0]) else {
			writeLog("\(namespaceName).querySelectionInfo unable to retrieve " +
				"firstItem. Database is empty or corrupt")
			throw ModelError.evaluationFailed
		}

		var next = BufferUInt8()
		let container = try DataContainer(from: firstItem.buffer, next: &next)
		return SelectionInfo(firstItem: container, dataKeys: keys)
	}

	//----------------------------------------------------------------------------
	// queryNextKeysInfo
	//  this is to retrieve the next 'count' number of containers
	private func queryNextKeysInfo(session: DbSession,
	                               selection: Selection) throws -> SelectionInfo {
		var dataKeys = [DataKey]()

		// create a read cursor
		let cursor = try session.dataTable.transaction().cursor()

		// start from requested position and get the extent
		guard let firstItem = (nextPrefetchKey == nil) ?
			try cursor.first() : try cursor.set(key: nextPrefetchKey!) else
		{
			writeLog("\(namespaceName).querySelectionInfo unable to retrieve " +
				"firstItem. Database is empty or corrupt")
			throw ModelError.evaluationFailed
		}

		var next = BufferUInt8()
		let container = try DataContainer(from: firstItem.buffer, next: &next)
		dataKeys.append(firstItem.key)

		// get the keys for the remaining items
		for _ in 1..<selection.count {
			if let item = try cursor.next() {
				dataKeys.append(item.key)
			} else if selection.wrapAround {
				if let item = try cursor.first() {
					dataKeys.append(item.key)
				}
			} else {
				// there are no more entries
				nextPrefetchKey = nil
				break
			}
		}

		// get the next key if we didn't hit the end
		if dataKeys.count == selection.count {
			nextPrefetchKey = try cursor.next()?.key
		}

		return SelectionInfo(firstItem: container, dataKeys: dataKeys)
	}

	//============================================================================
	// rebuildDatabase
	//
	private func rebuildDatabase(taskGroup: TaskGroup) throws {
		// setup the source
		guard let source = self.source else {
			writeLog("\(namespaceName): source is undefined")
			throw ModelError.setupFailed
		}

		try source.setupData(taskGroup: taskGroup)

		// re-create database
		writeLog("", level: .status, trailing: "*")
		writeLog("Database begin build: \(namespaceName)", level: .status)
		if provider.databaseExists {
			try provider.removeDatabase()
		}

		// start the timer
		let start = Date()

		// build source index list
		var sourceIndexes = Array(UInt32(0)..<UInt32(source.count))
		if shuffle {
			writeLog("Shuffling source indexes", level: .status)
			sourceIndexes.shuffle()
		}

		// append the source items to the database
		if source.count > 0 {
			try append(taskGroup: taskGroup, source: source, sourceIndexes: sourceIndexes)
		} else {
			writeLog("\(source.namespaceName) reports 0 items", level: .warning)
		}

		// report build time
		let end = Date()
		writeLog("Added \(source.count) items to database: \(namespaceName)",
			level: .status, indent: 1)
		writeLog("Database complete. Elapsed time: " +
			String(timeInterval: end.timeIntervalSince(start)), level: .status)
		writeLog("", level: .status, trailing: "*")


		// open db
		if validateBuild {
			writeLog("Database begin validation: \(namespaceName)", level: .status)
			let session = try provider.open(mode: .readOnly)

			for i in 0..<source.count {
				do {
					try validate(session: session, source: source, taskGroup: taskGroup, key: i)
				} catch {
					writeLog("Database validation error at key: \(i)")
					raise(SIGINT)
				}
			}
			writeLog("Database validation complete", level: .status)
			writeLog("", level: .status, trailing: "*")
		}
	}
	
	//----------------------------------------------------------------------------
	// append
	public func append(taskGroup: TaskGroup, source: DataSource,
	                   sourceIndexes: [UInt32]) throws {
		// open db
		let session = try provider.open(mode: .readWrite)

		// get conformance extent
		// If there is an item already in the database, then get it's
		// extent for conformance, otherwise use the first data source item
		var firstItem: ModelLabeledContainer
		var uniformShape: Shape
		if let dbData = try session.dataTable.transaction().cursor().first() {
			var next = BufferUInt8()
			firstItem = try DataContainer(from: dbData.buffer, next: &next)
			uniformShape = firstItem.shape
		} else {
			// get the extents of the encoded item, because it might change format
			firstItem = try source.getItem(at: 0, taskGroup: taskGroup)
			uniformShape = try firstItem.encodedShape()
		}

		// initialize pending records list
		let itemsDataLimitCount =
			writeBufferSize / (uniformShape.elementCount * firstItem.dataType.size)
		let maxItemsFetchCount = min(min(1000, source.count), itemsDataLimitCount)

		// loop through data source items
		for i in stride(from: 0, to: source.count, by: maxItemsFetchCount) {
			let itemsFetchCount = min(maxItemsFetchCount, source.count - i)

			// create items buffer that does not do copy on write
			var records = [(data: [UInt8]?, error: Error?)](repeating: (nil, nil), count: itemsFetchCount)
			let items = records.withUnsafeMutableBufferPointer { $0 }

			//------------------------------------
			// helper so same code can be used for serial or concurrent execution
			let fetchItems: (Int) -> Void = {
				[unowned self, unowned taskGroup, unowned source] index in

				do {
					// get the container and serialize
					let idx = Int(sourceIndexes[index])
					let container = try source.getItem(at: idx, taskGroup: taskGroup)
					var outputBuffer = [UInt8]()
					try container.serialize(to: &outputBuffer)

					// check conformity now the extent is guaranteed to be available
					switch self.requireUniform {
					case .extent:
						if container.shape != uniformShape {
							let dims = container.shape.extent
							if let uri = container.uri {
								self.writeLog("Uniform Extent requirement failed for \(uri.absoluteString)" +
									" Found: \(dims) expected: \(uniformShape.extent) ")
							} else {
								self.writeLog("Uniform Extent requirement failed. " +
									" Found: \(dims) expected: \(uniformShape.extent) ")
							}
							throw ModelError.uniformDataRequirementFailed
						}

					case .numberOfChannels:
						if container.shape.channels != uniformShape.channels {
							if let uri = container.uri {
								self.writeLog("Uniform numberOfChannels requirement failed for " +
									"\(uri.absoluteString) expected channels: \(uniformShape.channels)" +
									" found: \(container.shape.channels)")
							} else {
								self.writeLog("Uniform numberOfChannels requirement failed. expected" +
									" channels: \(uniformShape.channels) found: \(container.shape.channels)")
							}
							throw ModelError.uniformDataRequirementFailed
						}

					case .any: break
					}

					// assign valid item
					items[index].data = outputBuffer

				} catch {
					self.writeLog(String(describing: error))
					items[index].error = error
				}
			}

			//--------------------------------
			// execute
			if model.concurrencyMode == .concurrent {
				DispatchQueue.concurrentPerform(iterations: itemsFetchCount) { index in
					fetchItems(index)
				}
			} else { // .serial
				for index in 0..<itemsFetchCount {
					fetchItems(index)
				}
			}

			//--------------------------------
			// commit non nil items
			let validItems = records.flatMap { $0.data }
			
			if willLog(level: .diagnostic) {
				diagnostic("Committing: \(validItems.count)",
					categories: [.setup, .setupForward])
			}

			if !validItems.isEmpty {
				let txnData =	try session.dataTable.transaction(mode: .readWrite)
				_ = try txnData.append(data: validItems)
				try txnData.commit()
			}
			
			// throw first error if there is one
			let errors = records.flatMap { $0.error }
			guard errors.count == 0 else { throw errors[0] }
		}
	}

	//----------------------------------------------------------------------------
	// validate
	private func validate(session: DbSession, source: DataSource,
	                      taskGroup: TaskGroup, key: DataKey) throws {
		let txnReadData = try session.dataTable.transaction(mode: .readOnly)
		let cursor = try txnReadData.cursor()
		let dbData = try cursor.set(key: key)!
		var next   = BufferUInt8()
		let other  = try DataContainer(from: dbData.buffer, next: &next)
		let original = try source.getItem(at: other.sourceIndex, taskGroup: taskGroup)
		try original.verifyEqual(to: other)
	}

} // Database




