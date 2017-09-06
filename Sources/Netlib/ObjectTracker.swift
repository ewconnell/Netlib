//******************************************************************************
//  Created by Edward Connell on 1/10/17
//  Copyright © 2016 Connell Research. All rights reserved.
//
import Dispatch
import Foundation

#if os(Linux)
import Glibc
#endif

public protocol ObjectTracking : class {
	var trackingId: Int { get }
}

// singleton
public var objectTracker = ObjectTracker()

final public class ObjectTracker
{
	public struct ItemInfo {
		weak var object: Properties?
		let typeName: String
		let supplementalInfo: String
		var isStatic: Bool
	}

	// properties
	private let queue = DispatchQueue(label: "ObjectTracker.queue")
	public let counter = AtomicCounter()
	public var debuggerRegisterBreakId = -1
	public var debuggerRemoveBreakId = -1
	public private(set) var activeObjects = [Int : ItemInfo]()
	public var hasActiveObjects: Bool { return !activeObjects.isEmpty }

	//----------------------------------------------------------------------------
	// activeObjectsInfo
	public func getActiveObjectInfo(includeStatics: Bool = false) -> String {
		var result = "\n"
		var activeCount = 0
		for objectId in (activeObjects.keys.sorted { $0 < $1 }) {
			let info = activeObjects[objectId]!
			if includeStatics || !info.isStatic {
				result += getObjectDescription(id: objectId, info: info) + "\n"
				activeCount += 1
			}
		}
		if activeCount > 0 {
			result += "\nObjectTracker contains \(activeCount) live objects\n"
		}
		return result
	}

	// getObjectDescription
	private func getObjectDescription(id: Int, info: ItemInfo) -> String {
		var description = "[\(info.typeName)(\(id))"
		if info.supplementalInfo.isEmpty {
			description += "]"
		} else {
			description += " \(info.supplementalInfo)]"
		}

		if let propObject = info.object {
			description += " path: \(propObject.namePath)"
		}
		return description
	}

	//----------------------------------------------------------------------------
	// register(object:
	public func register(object: Properties, info: String = "") -> Int {
		let id = counter.increment()
		#if ENABLE_TRACKING
			register(id: id, info:
				ItemInfo(object: object, typeName: object.typeName,
				         supplementalInfo: info, isStatic: false))
		#endif
		return id
	}

	//----------------------------------------------------------------------------
	// register(type:
	public func register<T>(type: T, info: String = "") -> Int {
		let id = counter.increment()
		#if ENABLE_TRACKING
			register(id: id, info:
				ItemInfo(object: nil,
					       typeName: String(describing: Swift.type(of: T.self)),
				         supplementalInfo: info, isStatic: false))
		#endif
		return id
	}
	
	//----------------------------------------------------------------------------
	// register
	private func register(id: Int, info: ItemInfo) {
		queue.sync {
			if id == debuggerRegisterBreakId {
				print("ObjectTracker debug break for id(\(debuggerRegisterBreakId))")
				raise(SIGINT)
			}
			activeObjects[id] = info
		}
	}
	
	//----------------------------------------------------------------------------
	// markStatic
	public func markStatic(trackingId: Int) {
		#if ENABLE_TRACKING
			_ = queue.sync { activeObjects[trackingId]!.isStatic = true }
		#endif
	}
	
	//----------------------------------------------------------------------------
	// remove
	public func remove(trackingId: Int) {
		#if ENABLE_TRACKING
			_ = queue.sync {
				if trackingId == debuggerRemoveBreakId {
					print("ObjectTracker debug break remove for id(\(debuggerRemoveBreakId))")
					raise(SIGINT)
				}
				activeObjects.removeValue(forKey: trackingId)
			}
		#endif
	}
} // ObjectTracker
