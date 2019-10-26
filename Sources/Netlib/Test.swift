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
final public class Test : ModelElementContainerBase, Copyable, InitHelper {
	//----------------------------------------------------------------------------
	// properties
	public var topScores = [String : TestScore]()        { didSet{onSet("topScores")} }
	public var topScoreItems = [String : ModelElement]() { didSet{onSet("topScoreItems")}}

	//----------------------------------------------------------------------------
	// addAccessors
	public override func addAccessors() {
		super.addAccessors()
		addAccessor(name: "topScores",
		            get: { [unowned self] in self.topScores },
		            set: { [unowned self] in self.topScores = $0 })
		addAccessor(name: "topScoreItems",
		            get: { [unowned self] in self.topScoreItems },
		            set: { [unowned self] in self.topScoreItems = $0 })
	}

	//----------------------------------------------------------------------------
	// setup
	//   Function elements are containers with flexible in/out connector
	// arrangements.
	public override func setup(taskGroup: TaskGroup) throws {
		// templates and connector definitions are applied by the base classes
		try super.setup(taskGroup: taskGroup)

		// validate item types
		for item in items {
			if !(item is ModelElementContainer) {
				writeLog("Test item \(item.namespaceName) must be an ModelElementContainer")
				throw ModelError.setupFailed
			}
		}
		
		// the final container makes the connections
		try connectElements()
	}
}

//==============================================================================
// TestScore
public struct TestScore : AnyConvertible {
	// initializers
	public init(epoch: Double, testError: Double) {
		self.epoch = epoch
		self.testError = testError
	}

	public init(any: Any) throws {
		switch any {
		case is TestScore: self = any as! TestScore
		case let value as String:
			let params = value.components(separatedBy: ",").map {
				$0.components(separatedBy: ":")[1]
			}
			guard let epoch = Double(params[0]), let testError = Double(params[1]) else {
				throw PropertiesError.conversionFailed(type: TestScore.self, value: any)
			}
			self = TestScore(epoch: epoch, testError: testError)

		default: throw PropertiesError.conversionFailed(type: TestScore.self, value: any)
		}
	}

	public var asAny: Any {
		return "epoch:\(epoch), testError:\(testError)"
	}

	// properties
	var epoch: Double
	var testError: Double
}
