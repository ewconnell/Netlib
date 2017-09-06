//******************************************************************************
//  Created by Edward Connell on 6/4/16
//  Copyright © 2016 Connell Research. All rights reserved.
//

//==============================================================================
// DbProvider
public protocol DbProvider: ModelObject {
	// properties
	var connection: String { get set }
	var dataDir: String { get set }
	var databaseExists: Bool { get }
	var isMemoryMapped: Bool { get }
	var mapSize: Int { get set }

	// management
	func open(mode: DbAccessMode) throws -> DbSession
	func removeDatabase() throws
}

//------------------------------------------------------------------------------
// DbSession
public protocol DbSession : class {
	var dataTable: DbDataTable { get }
	func transaction() throws -> DbTransaction
}

public let dataTableName = "data"

//------------------------------------------------------------------------------
// DataKey
public typealias DataKey = Int
public extension DataKey
{
	static var unknownKey: DataKey { get { return -1 } }
}

//------------------------------------------------------------------------------
public enum DbAccessMode : String, EnumerableType {
	case readOnly, readWrite
}

//------------------------------------------------------------------------------
public enum DbError : Error {
	case commitFailed
	case createFailed(String)
	case openFailed
	case removeFailed
	case transactionFailed(String)
	case databaseProviderError(String)
	case keyNotFound
}

//==============================================================================
// DbData
public struct DbData {
	var key   : DataKey
	var buffer: BufferUInt8
	var data  : [UInt8] { get { return [UInt8](buffer)} }
}

//==============================================================================
// DbDataCursor
public protocol DbDataCursor : class {
	func first() throws -> DbData?
	func next()  throws -> DbData?
	func last()  throws -> DbData?
	func set(key: DataKey) throws -> DbData?
}

//==============================================================================
// DbTransaction
public protocol DbTransaction : class {
	func abort()
	func commit() throws
	func count() throws -> Int
}

//==============================================================================
// DbDataTransaction
public protocol DbDataTransaction : DbTransaction {
	func cursor() throws -> DbDataCursor
	func append(data: [UInt8]) throws -> DataKey
	func append(data: [[UInt8]]) throws -> [DataKey]
	func remove(dataKey: DataKey) throws
	func update(dataKey: DataKey, data: [UInt8]) throws
}

//==============================================================================
// DbDataTable
public protocol DbDataTable : class {
	func count() throws -> Int
	func transaction(parent: DbTransaction?, mode: DbAccessMode) throws -> DbDataTransaction
}

extension DbDataTable {
	public func transaction(mode: DbAccessMode = .readOnly) throws -> DbDataTransaction {
		return try transaction(parent: nil, mode: mode)
	}
}






