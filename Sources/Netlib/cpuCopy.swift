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

//==============================================================================
// cpuCopyAs
//	This maps elements converting data types and stride arrangements
//
public func cpuCopy(from inData: DataView,
                    to outData: inout DataView,
                    normalizeInts: Bool = false) throws {
	assert(inData.elementCount == outData.elementCount)
	assert(inData.items == outData.items)
	assert(inData.channels == outData.channels)
	assert(inData.rows == outData.rows)
	assert(inData.cols == outData.cols)

	// work function
	func copyValues<T,U>(_ result: inout DataView, _ t: T.Type, _ u: U.Type) throws
		where T: AnyNumber, U: AnyNumber
	{
		// get buffer pointers
		let inBuff  = try inData.ro(type: T.self)
		let outBuff = try result.rw(type: U.self)
		if normalizeInts {
			inData.forEachIndex(with: result) {
				outBuff[$1] = U(norm: inBuff[$0])
			}
		} else {
			inData.forEachIndex(with: result) {
				outBuff[$1] = U(any: inBuff[$0])
			}
		}
	}

	// switch to correct dynamic case
	switch inData.dataType {
	case .real8U:
		switch outData.dataType {
		case .real8U:  try copyValues(&outData, UInt8.self, UInt8.self)
		case .real16U: try copyValues(&outData, UInt8.self, UInt16.self)
		case .real16I: try copyValues(&outData, UInt8.self, Int16.self)
		case .real32I: try copyValues(&outData, UInt8.self, Int32.self)
		case .real16F: try copyValues(&outData, UInt8.self, Float16.self)
		case .real32F: try copyValues(&outData, UInt8.self, Float.self)
		case .real64F: try copyValues(&outData, UInt8.self, Double.self)
		}
		
	case .real16U:
		switch outData.dataType {
		case .real8U:  try copyValues(&outData, UInt16.self, UInt8.self)
		case .real16U: try copyValues(&outData, UInt16.self, UInt16.self)
		case .real16I: try copyValues(&outData, UInt16.self, Int16.self)
		case .real32I: try copyValues(&outData, UInt16.self, Int32.self)
		case .real16F: try copyValues(&outData, UInt16.self, Float16.self)
		case .real32F: try copyValues(&outData, UInt16.self, Float.self)
		case .real64F: try copyValues(&outData, UInt16.self, Double.self)
		}
		
	case .real16I:
		switch outData.dataType {
		case .real8U:  try copyValues(&outData, Int16.self, UInt8.self)
		case .real16U: try copyValues(&outData, Int16.self, UInt16.self)
		case .real16I: try copyValues(&outData, Int16.self, Int16.self)
		case .real32I: try copyValues(&outData, Int16.self, Int32.self)
		case .real16F: try copyValues(&outData, Int16.self, Float16.self)
		case .real32F: try copyValues(&outData, Int16.self, Float.self)
		case .real64F: try copyValues(&outData, Int16.self, Double.self)
		}

	case .real32I:
		switch outData.dataType {
		case .real8U:  try copyValues(&outData, Int32.self, UInt8.self)
		case .real16U: try copyValues(&outData, Int32.self, UInt16.self)
		case .real16I: try copyValues(&outData, Int32.self, Int16.self)
		case .real32I: try copyValues(&outData, Int32.self, Int32.self)
		case .real16F: try copyValues(&outData, Int32.self, Float16.self)
		case .real32F: try copyValues(&outData, Int32.self, Float.self)
		case .real64F: try copyValues(&outData, Int32.self, Double.self)
		}

	case .real16F:
		switch outData.dataType {
		case .real8U:  try copyValues(&outData, Float16.self, UInt8.self)
		case .real16U: try copyValues(&outData, Float16.self, UInt16.self)
		case .real16I: try copyValues(&outData, Float16.self, Int16.self)
		case .real32I: try copyValues(&outData, Float16.self, Int32.self)
		case .real16F: try copyValues(&outData, Float16.self, Float16.self)
		case .real32F: try copyValues(&outData, Float16.self, Float.self)
		case .real64F: try copyValues(&outData, Float16.self, Double.self)
		}
		
	case .real32F:
		switch outData.dataType {
		case .real8U:  try copyValues(&outData, Float.self, UInt8.self)
		case .real16U: try copyValues(&outData, Float.self, UInt16.self)
		case .real16I: try copyValues(&outData, Float.self, Int16.self)
		case .real32I: try copyValues(&outData, Float.self, Int32.self)
		case .real16F: try copyValues(&outData, Float.self, Float16.self)
		case .real32F: try copyValues(&outData, Float.self, Float.self)
		case .real64F: try copyValues(&outData, Float.self, Double.self)
		}
		
	case .real64F:
		switch outData.dataType {
		case .real8U:  try copyValues(&outData, Double.self, UInt8.self)
		case .real16U: try copyValues(&outData, Double.self, UInt16.self)
		case .real16I: try copyValues(&outData, Double.self, Int16.self)
		case .real32I: try copyValues(&outData, Double.self, Int32.self)
		case .real16F: try copyValues(&outData, Double.self, Float16.self)
		case .real32F: try copyValues(&outData, Double.self, Float.self)
		case .real64F: try copyValues(&outData, Double.self, Double.self)
		}
	}
}

