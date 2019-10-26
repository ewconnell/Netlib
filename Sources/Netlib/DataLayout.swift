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

public enum DataLayout : Int, AnyConvertible, BinarySerializable {

	case vector, matrix, nhwc, nchw, nchw_vector_c

	//----------------------------------------------------------------------------
	// identify axis
	public var channelsAxis: Int { return [0, 0, 3, 1, 1][self.rawValue] }
	public var rowsAxis: Int { return [0, 0, 1, 2, 2][self.rawValue] }
	public var colsAxis: Int { return [0, 1, 2, 3, 3][self.rawValue] }

	//----------------------------------------------------------------------------
	// AnyConvertible protocol
	public init(any: Any) throws {
		guard let value = any as? String else {
			throw PropertiesError
				.conversionFailed(type: LogLevel.self, value: any)
		}

		switch value.lowercased() {
		case "vector": self = .vector
		case "matrix": self = .matrix
		case "nhwc": self = .nhwc
		case "nchw": self = .nchw
		case "nchw_vector_c" : self = .nchw_vector_c
		default:
			throw PropertiesError.conversionFailed(type: DataLayout.self, value: any)
		}
	}

	public init?(string: String) { try? self.init(any: string) }

	public var asAny: Any {
		return ["vector", "matrix", "nhwc", "nchw", "nchw_vector_c"][self.rawValue]
	}
}
