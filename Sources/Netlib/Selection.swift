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
