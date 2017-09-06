//******************************************************************************
//  Created by Edward Connell on 1/24/17
//  Copyright © 2016 Connell Research. All rights reserved.
//
import Dispatch

public final class AtomicCounter {
	public init(value: Int = 0) { counter = value }
	
	public func increment() -> Int {
		return mutex.fastSync {
			counter += 1;
			return counter
		}
	}

	public var value: Int {
		get { return mutex.fastSync { counter } }
		set { return mutex.fastSync { counter = newValue } }
	}

	// properties
	private var counter: Int
	private let mutex = PThreadMutex()
}
