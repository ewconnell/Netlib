//******************************************************************************
//  Created by Edward Connell on 8/2/17
//  Copyright © 2016 Connell Research. All rights reserved.
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
