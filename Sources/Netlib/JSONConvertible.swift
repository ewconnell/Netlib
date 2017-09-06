//******************************************************************************
//  Created by Edward Connell on 5/1/16
//  Copyright © 2016 Connell Research. All rights reserved.
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



