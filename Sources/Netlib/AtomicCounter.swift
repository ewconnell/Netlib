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
