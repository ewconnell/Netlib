//******************************************************************************
//  Created by Edward Connell on 8/21/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
import Foundation

public struct Selection
{
	public init() { id = -1; count = 0; keys = nil; wrapAround = false }
	public init(count: Int, wrapAround: Bool = false) {
		self.id = Selection.counter.increment()
		self.count = count
		self.keys = nil
		self.wrapAround = wrapAround
	}

	public init(keys: [DataKey]) {
		self.id = Selection.counter.increment()
		self.keys = keys
		self.count = keys.count
		self.wrapAround = false
	}

	//----------------------------------------------------------------------------
	// properties
	private static var counter = AtomicCounter()
	public let count: Int
	public let id: Int
	public let keys: [DataKey]?
	public let wrapAround: Bool
	public var usesSpecificKeys: Bool { return keys != nil }
	
	// functions
	public func next() -> Selection {
		assert(keys == nil, "next cannot be used when keys are specified")
		return Selection(count: count, wrapAround: wrapAround)
	}

	public static func ==(lhs: Selection, rhs: Selection) -> Bool {
		return lhs.id == rhs.id
	}
	public static func !=(lhs: Selection, rhs: Selection) -> Bool {
		return !(lhs.id == rhs.id)
	}
}
