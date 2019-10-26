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
import Lmdb

#if os(Linux)
import Glibc
#endif

//==============================================================================
// DataSource
//  The Database element uses a specified DataSource to get items when
// the database is being built
//
public protocol DataSource : ModelObject {
	// properties
	var count: Int { get }
	var legend: [[String : String]]? { get set }
	var maxCount: Int { get set }

	// random access for multiple threads
	func getItem(at index: Int, taskGroup: TaskGroup) throws -> ModelLabeledContainer

	// this is a separate setup called by the Database object
	// only when the database needs rebuilding
	func setupData(taskGroup: TaskGroup) throws
}

extension DataSource {
	public func getItem(at index: Int) throws -> ModelLabeledContainer {
		return try getItem(at: index, taskGroup: model!.tasks)
	}
}

//==============================================================================
// DataSourceBase
//
open class DataSourceBase : ModelObjectBase, DefaultPropertiesContainer {
	//----------------------------------------------------------------------------
	// properties
	public var defaultTypeIndex = [String : [String : ModelDefault]]()
	public var legend: [[String : String]]?

	public var defaults: [Default]? {
		didSet{
			onDefaultsChanged();
			onSet("defaults")
		}}

	public var defaultValues: [String : String]? {
		didSet {
			onDefaultValuesChanged(oldValue: oldValue);
			onSet("defaultValues")
		}
	}

	public func onDefaultsChanged() {
		rebuildDefaultTypeIndex()
	}

	//----------------------------------------------------------------------------
	// addAccessors
	public override func addAccessors() {
		super.addAccessors()
		addAccessor(name: "defaults",
		            get: { [unowned self] in self.defaults },
		            set: { [unowned self] in self.defaults = $0 })
		addAccessor(name: "defaultValues", lookup: .noLookup,
		            get: { [unowned self] in self.defaultValues },
		            set: { [unowned self] in self.defaultValues = $0 })
	}

	//----------------------------------------------------------------------------
	// lookupDefault
	public override func lookupDefault(typePath: String, propPath: String) -> ModelDefault? {
		return lookup(typePath: typePath, propPath: propPath) ??
			parent?.lookupDefault(typePath: typePath, propPath: propPath)
	}
}
