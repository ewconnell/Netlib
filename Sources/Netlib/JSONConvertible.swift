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

//------------------------------------------------------------------------------
// JsonConvertible
//
public protocol JsonConvertible {
	func asJson(after modelVersion: Int, include: SelectAnyOptions,
	            options: JSONSerialization.WritingOptions) throws -> String
	
	func writeJson(to stream: OutputStream,
	               after modelVersion: Int, include: SelectAnyOptions,
	               options: JSONSerialization.WritingOptions) throws -> Int

	func update(fromJson string: String) throws
	func update(fromJson stream: InputStream) throws
}

public enum JsonConvertibleError: Error {
	case stringEncodingError
	case streamEncodingError(NSError)
}

extension JsonConvertible where Self: ModelObject
{
	//----------------------------------------------------------------------------
	// json text
	public func asJson(after modelVersion: Int = 0,
	                   include: SelectAnyOptions = [.storage],
	                   options: JSONSerialization.WritingOptions = []) throws -> String
	{
		// getting the Function state should always be valid, so any possible
		// exceptions are a programming error, therefore we don't rethrow
		let selected = selectAny(after: modelVersion, include: include) ?? AnySet()
		let data = try JSONSerialization.data(withJSONObject: selected, options: options)
		return String(data: data, encoding: .utf8)!
	}
	
	//----------------------------------------------------------------------------
	// json stream
	public func writeJson(to stream: OutputStream, after modelVersion: Int = 0,
	                      include: SelectAnyOptions = [.storage],
	                      options: JSONSerialization.WritingOptions = []) throws -> Int
	{
		// getting the Function state should always produce valid input
		var selected = selectAny(after: modelVersion, include: include) ?? AnySet()

		#if os(Linux)
			let written = try JSONSerialization
				.writeJSONObject(selected, toStream: stream, options: options)
		#else
			var error: NSError?
			let written = JSONSerialization
				.writeJSONObject(selected, to: stream, options: options, error: &error)
			if error != nil { throw JsonConvertibleError.streamEncodingError(error!)	}
		#endif

		return written
	}
	
	//----------------------------------------------------------------------------
	// json text
	public func update(fromJson string: String) throws {
		// encode the string to a Data object for json
		guard let data = string.data(using: .utf8) else {
			throw JsonConvertibleError.stringEncodingError
		}
		try updateAny(with: JSONSerialization.jsonObject(with: data) as? AnySet)
	}
	
	//----------------------------------------------------------------------------
	// json stream
	public func update(fromJson stream: InputStream) throws	{
		try updateAny(with: JSONSerialization.jsonObject(with: stream) as? AnySet)
	}
}



