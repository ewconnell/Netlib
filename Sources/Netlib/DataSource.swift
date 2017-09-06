//******************************************************************************
//  Created by Edward Connell on 4/25/16
//  Copyright © 2016 Connell Research. All rights reserved.
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
