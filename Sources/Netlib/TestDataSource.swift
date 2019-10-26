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

final public class TestDataSource : DataSourceBase, DataSource, InitHelper {
	//----------------------------------------------------------------------------
	// properties
	//	If dataItem is set, then only itemCount and defaults are used
	public var channelFormat = ChannelFormat.any   { didSet{onSet("channelFormat")}}
	public var colMajor = false                    { didSet{onSet("colMajor")} }
	public var containerTemplate = DataContainer() { didSet{onSet("containerTemplate")} }
	public var dataLayout = DataLayout.nchw        { didSet{onSet("dataLayout")}}
	public var dataType: DataType = .real32F       { didSet{onSet("dataType")} }
	public var indexBy = IndexBy.values            { didSet{onSet("indexBy")} }
	public var itemCount = 0                       { didSet{onSet("itemCount")} }
	public var itemExtent: [Int]?                  { didSet{onSet("itemExtent")} }
	public var maxCount = Int.max                  { didSet{onSet("maxCount")} }
	public var normalizeIndex = false              { didSet{onSet("normalizeIndex")}}
	public var uriData: Uri?                       { didSet{onSet("uriData")} }

	// count
	public var count: Int {
		return min(itemCount, maxCount)
	}

	//----------------------------------------------------------------------------
	// types
	public enum IndexBy : String, EnumerableType { case values, channels }
	
	//----------------------------------------------------------------------------
	// addAccessors
	public override func addAccessors() {
		super.addAccessors()
		addAccessor(name: "channelFormat",
		            get: { [unowned self] in self.channelFormat },
		            set: { [unowned self] in self.channelFormat = $0 })
		addAccessor(name: "colMajor",
		            get: { [unowned self] in self.colMajor },
		            set: { [unowned self] in self.colMajor = $0 })
		addAccessor(name: "containerTemplate",
		            get: { [unowned self] in self.containerTemplate },
		            set: { [unowned self] in self.containerTemplate = $0 })
		addAccessor(name: "dataLayout",
			          get: { [unowned self] in self.dataLayout },
			          set: { [unowned self] in self.dataLayout = $0 })
		addAccessor(name: "dataType",
		            get: { [unowned self] in self.dataType },
		            set: { [unowned self] in self.dataType = $0 })
		addAccessor(name: "indexBy",
		            get: { [unowned self] in self.indexBy },
		            set: { [unowned self] in self.indexBy = $0 })
		addAccessor(name: "itemCount",
		            get: { [unowned self] in self.itemCount },
		            set: { [unowned self] in self.itemCount = $0 })
		addAccessor(name: "itemExtent",
		            get: { [unowned self] in self.itemExtent },
		            set: { [unowned self] in self.itemExtent = $0 })
		addAccessor(name: "maxCount",
		            get: { [unowned self] in self.maxCount },
		            set: { [unowned self] in self.maxCount = $0 })
		addAccessor(name: "normalizeIndex",
		            get: { [unowned self] in self.normalizeIndex },
		            set: { [unowned self] in self.normalizeIndex = $0 })
		addAccessor(name: "uriData",
		            get: { [unowned self] in self.uriData },
		            set: { [unowned self] in self.uriData = $0 })
	}

	//----------------------------------------------------------------------------
	// setupData
	//	If dataItem is set, then only itemCount and defaults are used
	public func setupData(taskGroup: TaskGroup) throws {
		// initialize data
		if let uri = self.uriData {
			containerTemplate.uri = uri
			return
		}

		//--------------------------------------------------------------------------
		// synthesize
		guard let itemExtent = itemExtent else {
			writeLog("itemExtent must be specified")
			throw ModelError.setupFailed
		}
		
		var shape = Shape(extent: itemExtent, layout: dataLayout,
		                  channelFormat: channelFormat, colMajor: colMajor)
		
		var data = DataView(shape: shape, dataType: dataType)
		
		// set the values --------------------------------------------------------
		switch indexBy {
		case .values:
			func setValues<T: AnyNumber>(_ t: T.Type) throws {
				// fill each channel with the sample number
				var index = 0
				try data.forEachMutableSample { (dst: inout MutableDataSample<T>) in
					let value = normalizeIndex ? T(norm: index) : T(any: index)
					for i in 0..<shape.channels { dst[i] = value }
					index += 1
				}
			}
			
			switch dataType {
			case .real8U : try setValues(UInt8.self)
			case .real16U: try setValues(UInt16.self)
			case .real16I: try setValues(Int16.self)
			case .real32I: try setValues(Int32.self)
			case .real16F: try setValues(Float16.self)
			case .real32F: try setValues(Float.self)
			case .real64F: try setValues(Double.self)
			}
			
		case .channels:
			func setValues<T: AnyNumber>(_ t: T.Type) throws {
				// fill each channel with the sample number
				let channels = Double(shape.channels)
				try data.forEachMutableSample { (dst: inout MutableDataSample<T>) in
					for i in 0..<shape.channels {
						dst[i] = normalizeIndex ?
							T(norm: Double(i + 1) / channels) :
							T(any: i)
					}
				}
			}
			
			switch dataType {
			case .real8U : try setValues(UInt8.self)
			case .real16U: try setValues(UInt16.self)
			case .real16I: try setValues(Int16.self)
			case .real32I: try setValues(Int32.self)
			case .real16F: try setValues(Float16.self)
			case .real32F: try setValues(Float.self)
			case .real64F: try setValues(Double.self)
			}
		}
		
		// assign to template and setup
		containerTemplate.data = data
	}
	
	//----------------------------------------------------------------------------
	// getItem
	public func getItem(at index: Int, taskGroup: TaskGroup) throws -> ModelLabeledContainer {
		assert(index < count, "Index out of range")
		return try containerTemplate.copyTemplate(index: index, taskGroup: taskGroup) { _ in }
	}
}
