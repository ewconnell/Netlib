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
import Lmdb

final public class LmdbProvider : ModelObjectBase, DbProvider {
	//----------------------------------------------------------------------------
	// properties
	public var connection = ""                     { didSet{onSet("connection")} }
	public var dataDir = "~/data"                  { didSet{onSet("dataDir")} }
	public var mapSize = 10.GB	                   { didSet{onSet("mapSize")} }

	public var isMemoryMapped: Bool { return true }
	public var databaseExists: Bool {
		return FileManager.default.fileExists(atPath: getPath())
	}

	//----------------------------------------------------------------------------
	// addAccessors
	public override func addAccessors() {
		super.addAccessors()
		addAccessor(name: "connection",
		            get: { [unowned self] in self.connection },
		            set: { [unowned self] in self.connection = $0 })
		addAccessor(name: "dataDir",
		            get: { [unowned self] in self.dataDir },
		            set: { [unowned self] in self.dataDir = $0 })
		addAccessor(name: "mapSize",
		            get: { [unowned self] in self.mapSize },
		            set: { [unowned self] in self.mapSize = $0 })
	}

	//----------------------------------------------------------------------------
	// getPath
	fileprivate func getPath() -> String {
		let dirPath = NSString(string: dataDir).expandingTildeInPath
		let url = URL(fileURLWithPath: dirPath, isDirectory: true)
			.appendingPathComponent(connection)
		return url.path
	}

	//----------------------------------------------------------------------------
	// open
	public func open(mode: DbAccessMode) throws -> DbSession {
		// create if needed
		if !databaseExists {
			if mode == .readWrite {
				// create data folder
				let dbPath = getPath()
				try FileManager.default.createDirectory(
					atPath: dbPath, withIntermediateDirectories: true, attributes: nil)
				if willLog(level: .status) {
					writeLog("Creating database in: \(dbPath)", level: .status)
				}

				// create the tables
				let session = try LmdbSession(provider: self, mode: .readWrite)
				try session.dataTable.transaction(mode: .readWrite).commit()

			} else {
				writeLog("Database does not exist: \(connection)")
				throw DbError.openFailed
			}
		}

		return try LmdbSession(provider: self, mode: mode)
	}

	// removeDatabase
	public func removeDatabase() throws {
		if databaseExists {
			try FileManager.default.removeItem(atPath: getPath())
		}
	}
}

//==============================================================================
// LmdbSession
public final class LmdbSession : DbSession, ObjectTracking {
	// initializers
	public init(provider: LmdbProvider, mode: DbAccessMode) throws {
		// open the environment
		assert(!LmdbSession.isOpen,
			"LMDB does not allow multiple simultaneous opens per process")
		self.provider = provider
		var temp: OpaquePointer?
		try mdbCheck(status: mdb_env_create(&temp))
		model = temp!
		try mdbCheck(status: mdb_env_set_mapsize(model, provider.mapSize))

		// there is 1 table right now
		try mdbCheck(status: mdb_env_set_maxdbs(model, 1))

		var flags = UInt32(MDB_NOTLS)
		if mode == .readOnly { flags |= UInt32(MDB_RDONLY) }
		try mdbCheck(status: mdb_env_open(model, provider.getPath(), flags, 0o664))
		LmdbSession.isOpen = true

		trackingId = objectTracker.register(type: self)
	}

	deinit {
		mdb_env_close(model)
		LmdbSession.isOpen = false
		objectTracker.remove(trackingId: trackingId)
	}

	//----------------------------------------------------------------------------
	// properties
	fileprivate let provider: DbProvider
	fileprivate let model: OpaquePointer
	public private(set) var trackingId = 0
	private static var isOpen = false

	// tables
	public lazy var dataTable: DbDataTable =
		{ [unowned self] in LmdbDataTable(session: self) }()

	//----------------------------------------------------------------------------
	// transaction
	//   used as parent for group r/w transactions
	public func transaction() throws -> DbTransaction {
		return try LmdbTransaction(session: self)
	}
}


//-----------------------------------
// seek
// This is used by the other functions to reposition the cursor
private func seek<KeyT>(cursor: OpaquePointer, key: KeyT,
                       op: MDB_cursor_op) throws -> DbData? {
	var dataKey = key
	var mdb_key = MDB_val(mv_size: MemoryLayout<KeyT>.size, mv_data: &dataKey)
	var mdb_value = MDB_val()

	// move the cursor
	let status = mdb_cursor_get(cursor, &mdb_key, &mdb_value, op)
	if status == MDB_NOTFOUND {
		return nil
	} else {
		try mdbCheck(status: status)
	}

	// return the record at the new position
	let returnKey = mdb_key.mv_data.bindMemory(to: DataKey.self, capacity: 1)[0]
	let dataBuffer = UnsafeBufferPointer(
		start: mdb_value.mv_data.bindMemory(to: UInt8.self, capacity: mdb_value.mv_size),
		count: mdb_value.mv_size)
	return DbData(key: returnKey, buffer: dataBuffer)
}

//==============================================================================
// LmdbDataCursor
public final class LmdbDataCursor : DbDataCursor, ObjectTracking {
	// initializers
	init(transaction: LmdbTransaction) throws {
		self.transaction = transaction
		var p: OpaquePointer?
		try mdbCheck(status: mdb_cursor_open(transaction.txn, transaction.dbi, &p))
		self.cursor = p
		trackingId = objectTracker.register(type: self)
	}

	deinit {
		mdb_cursor_close(cursor)
		objectTracker.remove(trackingId: trackingId)
	}

	//----------------------------------------------------------------------------
	// properties
	private var transaction: LmdbTransaction
	private let cursor: OpaquePointer!
	public private(set) var trackingId = 0

	//-----------------------------------
	// first
	public func first() throws -> DbData? {
		return try seek(cursor: cursor, key: DataKey.unknownKey, op: MDB_FIRST)
	}

	//-----------------------------------
	// last
	public func last() throws -> DbData? {
		return try seek(cursor: cursor, key: DataKey.unknownKey, op: MDB_LAST)
	}

	//-----------------------------------
	// next
	public func next() throws -> DbData? {
		return try seek(cursor: cursor, key: DataKey.unknownKey, op: MDB_NEXT)
	}

	//-----------------------------------
	// set
	public func set(key: DataKey) throws -> DbData? {
		return try seek(cursor: cursor, key: key, op: MDB_SET)
	}
}

//==============================================================================
// LmdbTransaction
public class LmdbTransaction : DbTransaction, ObjectTracking {
	// group transaction initializer
	init(session: LmdbSession) throws {
		self.session = session
		self.txnParent = nil
		self.mode = .readWrite
		try mdbCheck(status: mdb_txn_begin(session.model, nil, 0, &txn))

		var i: UInt32 = 0
		try mdbCheck(status: mdb_dbi_open(txn, nil, 0, &i))
		self.dbi = i
		trackingId = objectTracker.register(type: self)
	}

	// table transaction initializer
	init(session: LmdbSession, parent: LmdbTransaction?, flags: UInt32,
	     tableName: String, mode: DbAccessMode) throws {
		// set environment and parent transaction handle if there is one
		self.session = session
		self.txnParent = parent?.txn
		self.mode = mode

		let rwMode    = (mode == .readOnly) ? UInt32(MDB_RDONLY) : 0
		let openFlags = (mode == .readWrite) ? (flags | UInt32(MDB_CREATE)) : flags

		try mdbCheck(status: mdb_txn_begin(session.model, txnParent, rwMode, &txn))

		var i: UInt32 = 0
		try mdbCheck(status: mdb_dbi_open(txn, tableName, openFlags, &i))
		self.dbi = i
	}

	deinit {
		// txn will be nil if the transaction was committed
		if let txn = self.txn {
			if mode == .readWrite {
				session.provider.diagnostic(
					"aborting transaction", categories: [.setup, .evaluate])
			}
			mdb_txn_abort(txn)
		}
		objectTracker.remove(trackingId: trackingId)
	}

	//----------------------------------------------------------------------------
	// properties
	private let session: LmdbSession
	private let txnParent: OpaquePointer?
	fileprivate var txn: OpaquePointer?
	fileprivate let dbi: UInt32
	private let mode: DbAccessMode
	public private(set) var trackingId = 0

	public func count() throws -> Int {
		var s = MDB_stat()
		try mdbCheck(status: mdb_stat(txn, dbi, &s))
		return Int(s.ms_entries)
	}

	//----------------------------------------------------------------------------
	// abort
	public func abort() {
		assert(txn != nil)
		mdb_txn_abort(txn)
		txn = nil
	}

	//----------------------------------------------------------------------------
	// commit
	public func commit() throws {
		// commit - commits and frees the transaction handle
		try mdbCheck(status: mdb_txn_commit(txn))
		txn = nil
	}

	//----------------------------------------------------------------------------
	// put
	public func put<KeyT>(key: KeyT, data: UnsafeRawPointer, size: Int,
	                      flags: Int32 = 0) throws {
		guard txn != nil else {
			throw DbError.transactionFailed("Cannot use transaction after commit")
		}
		var recordKey = key
		var mdb_key = MDB_val(mv_size: MemoryLayout<KeyT>.size, mv_data: &recordKey)

		let dataPtr = UnsafeMutableRawPointer(OpaquePointer(data))
		var mdb_value = MDB_val(mv_size: size, mv_data: dataPtr)
		try mdbCheck(status: mdb_put(txn, dbi, &mdb_key, &mdb_value, UInt32(flags)))
	}
}

//==============================================================================
// LmdbDataTransaction
public final class LmdbDataTransaction : LmdbTransaction, DbDataTransaction {
	// initializers
	public init(session: LmdbSession, parent: LmdbTransaction?,
	            tableName: String, mode: DbAccessMode) throws {

		try super.init(session: session, parent: parent,
		               flags: UInt32(MDB_INTEGERKEY),
		               tableName: tableName, mode: mode)
	}

	//----------------------------------------------------------------------------
	// properties
	var nextKey: DataKey?

	//----------------------------------------------------------------------------
	// append
	public func append(data: [UInt8]) throws -> DataKey {
		let key = try getNextKey()
		try put(key: key, data: data, size: data.count,
			      flags: MDB_APPEND | MDB_NOOVERWRITE)
		return key
	}

	//----------------------------------------------------------------------------
	// append
	public func append(data: [[UInt8]]) throws -> [DataKey] {
		var keys = [DataKey]()
		var key = try getNextKey()
		for i in 0..<data.count {
			try put(key: key, data: data[i], size: data[i].count,
				      flags: MDB_APPEND | MDB_NOOVERWRITE)
			keys.append(key)
			key += 1
		}
		return keys
	}

	//----------------------------------------------------------------------------
	// update
	public func update(dataKey: DataKey, data: [UInt8]) throws {
		try put(key: dataKey, data: data, size: data.count)
	}

	//----------------------------------------------------------------------------
	// remove
	public func remove(dataKey key: DataKey) throws {
		var dataKey = key
		var mdb_key = MDB_val(mv_size: MemoryLayout<DataKey>.size, mv_data: &dataKey)
		try mdbCheck(status: mdb_del(txn, dbi, &mdb_key, nil))
	}

	//----------------------------------------------------------------------------
	// cursor
	public func cursor() throws -> DbDataCursor {
		return try LmdbDataCursor(transaction: self)
	}

	//----------------------------------------------------------------------------
	// getNextKey
	private func getNextKey() throws -> DataKey {
		if nextKey == nil {
			if let last = try cursor().last() {
				nextKey = last.key + 1
			} else {
				nextKey = 0
			}
		} else {
			nextKey! += 1
		}
		return nextKey!
	}
}

//==============================================================================
// LmdbDataTable
public final class LmdbDataTable : DbDataTable {
	// initializer
	public init(session: LmdbSession) { self.session = session }

	//----------------------------------------------------------------------------
	// properties
	private unowned var session: LmdbSession

	//----------------------------------------------------------------------------
	// count
	public func count() throws -> Int {
		return try transaction(mode: .readOnly).count()
	}

	//----------------------------------------------------------------------------
	// transaction
	public func transaction(parent: DbTransaction?,
	                        mode: DbAccessMode) throws -> DbDataTransaction {
		return try LmdbDataTransaction(session: session,
		                               parent: parent as? LmdbTransaction,
		                               tableName: dataTableName, mode: mode)
	}
}

//==============================================================================
// helpers
func mdbCheck(status: CInt) throws	{
	if status != MDB_SUCCESS {
		throw DbError.databaseProviderError(String(cString: mdb_strerror(status)))
	}
}

