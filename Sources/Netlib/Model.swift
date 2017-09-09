//******************************************************************************
//  Created by Edward Connell on 3/31/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
import Foundation
import Dispatch

//==============================================================================
// Model
//	A model represents a root compute instance for a function graph
public final class Model :
	ModelElementContainerBase, XmlConvertible, JsonConvertible, Copyable, InitHelper {
	//-----------------------------------
	// initializers
	public required init() {
		super.init()

		// this is the root
		model = self
		currentLog = log

		// set context for member objects
		properties.values.forEach { $0.setContext() }
	}

	//-----------------------------------
	// root model initialized from URL
	public convenience init(contentsOf url: URL, logLevel: LogLevel = .status) throws {
		self.init()
		log.logLevel = logLevel
		try load(contentsOf: url)
	}

	deinit {
		self.waitForAsyncTasksToComplete()
	}

	//----------------------------------------------------------------------------
	// properties
	public var compute = ComputePlatform()                  { didSet{onSet("compute")}}
	public var concurrencyMode = ConcurrencyMode.concurrent { didSet{onSet("concurrencyMode")}}
	public var host: Uri?                                   { didSet{onSet("host")}}
	public var log = Log()                                  { didSet{onSet("log")}}
	public var tasks = TaskGroup()		                      { didSet{onSet("tasks")}}
	public var learnedParameters = [String : LearnedParameter]()
	private let setupOrder = [
		VersionKey, TypeNameKey, "host", "log",
		"concurrencyMode", "compute", "tasks"
	]

	//----------------------------------------------------------------------------
	// addAccessors
	public override func addAccessors() {
		super.addAccessors()
		addAccessor(name: "compute",
		            get: { [unowned self] in self.compute },
		            set: { [unowned self] in self.compute = $0 })
		addAccessor(name: "concurrencyMode",
		            get: { [unowned self] in self.concurrencyMode },
		            set: { [unowned self] in self.concurrencyMode = $0 })
		addAccessor(name: "host",
		            get: { [unowned self] in self.host },
		            set: { [unowned self] in self.host = $0 })
		addAccessor(name: "log",
		            get: { [unowned self] in self.log },
		            set: { [unowned self] in self.log = $0 })
		addAccessor(name: "tasks",
		            get: { [unowned self] in self.tasks },
		            set: { [unowned self] in self.tasks = $0 })
		addAccessor(name: TypeNameKey, lookup: .noLookup,
		            get: { [unowned self] in self.typeName },
		            set: { _ in })
		addAccessor(name: VersionKey, lookup: .noLookup,
		            get: { [unowned self] in self.version },
		            set: { [unowned self] in self.version = $0 })
	}
	
	//----------------------------------------------------------------------------
	// version
	private var modelVersion = 0
	private var versioningDisabledCount = 0
	private let versionMutex = PThreadMutex()

	// incrementVersion
	@discardableResult
	public func incrementVersion() -> Int {
		var prop: AnyProperty?
		let newVersion = versionMutex.fastSync { () -> Int in
			if versioningDisabledCount == 0 {
				modelVersion += 1
				prop = properties[VersionKey]
				prop!.version = modelVersion
				prop!.isDefault = false
			}
			return modelVersion
		}
		prop?.changed.raise(data: self)
		return newVersion
	}
	
	public override var version: Int {
		get { return versionMutex.fastSync { modelVersion } }
		set {
			var prop: AnyProperty!
			versionMutex.fastSync {
				modelVersion = newValue
				prop = properties[VersionKey]
				prop.version = modelVersion
				prop.isDefault = false
			}
			prop.changed.raise(data: self)
		}
	}
	
	public func incrementVersioningDisabledCount() {
		versionMutex.fastSync { versioningDisabledCount += 1 }
	}
	
	public func decrementVersioningDisabledCount() {
		versionMutex.fastSync { versioningDisabledCount -= 1 }
	}
	
	//----------------------------------------------------------------------------
	// waitForAsyncTasksToComplete
	//  Some objects have async prefetch tasks such as the Database element
	// this method is used to wait until they are all complete, usually
	// before destroying the Function
	// Calling it in deinit doesn't seem to work
	public let asyncTaskGroup = DispatchGroup()
	public func waitForAsyncTasksToComplete() { asyncTaskGroup.wait() }
	
	//----------------------------------------------------------------------------
	// copy(from other
	public override func copy(from other: Properties) {
		// set the version first, so it is properly reflected by other properties
		// as they are attached and setContext version stamps them
		version = other.model.version
		super.copy(from: other)
	}
	
	//----------------------------------------------------------------------------
	// load
	public func load(contentsOf url: URL) throws {
		// increment
		incrementVersion()
		
		try withVersioningDisabled {
			// set storage location of container to support relative URLs
			storage = Uri(url: url.deletingLastPathComponent())
			
			// load the update set
			let updates = try updateSet(fromXml: url)
			
			// get root type of external fragment
			guard let rootTypeName = updates[TypeNameKey] as? String else {
				writeLog("typename is missing from XML object definition")
				throw ModelError.loadFailed
			}
			
			// update model or replace contents of the items collection
			if rootTypeName == typeName {
				// the root of the external fragment is of type Model,
				// so update this object
				try update(fromXml: url)

			} else {
				// the root of the external fragment is not a model,
				// so replace this model's items collection
				let rootObject = try Create(typeName: rootTypeName)
				guard let element = rootObject as? ModelElementBase else {
					writeLog("root XML object must derive from ModelElementBase")
					throw ModelError.loadFailed
				}
				items.removeAll()
				items.append(element)
				try element.updateAny(with: updates)

				if willLog(level: .diagnostic) {
					diagnostic("Loaded external template \(element.namespaceName) into " +
						"\(namespaceName).items", categories: .setup)
				}
			}
		}
	}
	
	//----------------------------------------------------------------------------
	// updateAny
	public override func updateAny(with updates: AnySet?) throws {
		try withVersioningDisabled {
			// set the new model version first
			if let updateVersion = updates?[VersionKey] as? Int {
				version = updateVersion
			}
			try super.updateAny(with: updates)
		}
	}
	
	//----------------------------------------------------------------------------
	// createCodec
	// TODO: make this overridable through plugins
	public static func createCodec(for category: CodecType) throws -> Codec {
		switch category {
		case .image: return ImageCodec()
		case .data: return DataCodec()
		default: throw ModelError.codecTypeNotFound(category)
		}
	}
	
	//----------------------------------------------------------------------------
	// convenience function for the model Function
	public func setup() throws {
		try setup(taskGroup: tasks)
	}
	
	//----------------------------------------------------------------------------
	// setup
	public override func setup(taskGroup: TaskGroup) throws {
		// clear in case of multiple setup
		model.learnedParameters.removeAll(keepingCapacity: true)

		// setup this objects props in the specified order
		try setupOrder.forEach {
			let prop = properties[$0]!
			try prop.setup(taskGroup: taskGroup)
			prop.isExplicitlySetup = true
		}
		
		// setup base
		try super.setup(taskGroup: taskGroup)
		
		// connect
		try connectElements()
	}
	
	//----------------------------------------------------------------------------
	// train
	public func train(handler: TrainingCompletionHandler? = nil) throws {
		// make sure things are setup
		try setup()
		
		// find solvers
		let solvers = try self.filter { $0 is ModelSolver } as! [ModelSolver]
		guard solvers.count > 0 else {
			writeLog("\(namespaceName) no solvers were found")
			return
		}
		
		// if a completion handler is specified then solve async
		if let handler = handler {
			DispatchQueue.global().async {
				for solver in solvers {
					do {
						try solver.solve()
						handler(solver, nil)
					} catch {
						self.writeLog("\(self.namespaceName) - \(error)")
						handler(solver, error)
						break
					}
				}
			}
		} else {
			for solver in solvers {
				do {
					try solver.solve()
				} catch {
					writeLog(String(describing: error))
					throw error
				}
			}
		}
	}
}

//==============================================================================
// ModelError
public enum ModelError : Error {
	case rangeError(String)
	case setupFailed
	case loadFailed
	case evaluationFailed
	case error(String)
	case notImplemented(String)
	case argumentIsNil(String)
	case conversationFailed(String)
	case codecTypeNotFound(CodecType)
	case validationFailed
	case uniformDataRequirementFailed
	case namespaceError
}

public enum ConcurrencyMode : String, EnumerableType { case serial, concurrent }

