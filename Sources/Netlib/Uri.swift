//******************************************************************************
//  Created by Edward Connell on 4/11/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
import Foundation

final public class Uri : ModelObjectBase, Equatable, InitHelper {
	public convenience init(string: String, cacheDir: String? = nil,
	                        cacheFile: String? = nil) {
		self.init()
		self.string = string; onSet("string")
		if let cacheDir = cacheDir { self.cacheDir = cacheDir; onSet("cacheDir") }
		if let cacheFile = cacheFile { self.cacheFile = cacheFile; onSet("cacheFile") }
	}

	public convenience init(filePath: String) {
		self.init(string: URL(fileURLWithPath: filePath).absoluteString)
	}

	public convenience init(url: URL) {	self.init(string: url.absoluteString) }

	//----------------------------------------------------------------------------
	// properties
	//	This are string urls in order to avoid a recursion problem
	public var cacheDir: String?                    { didSet{onSet("cacheDir")} }
	public var cacheFile: String?                   { didSet{onSet("cacheFile")} }
	public var dataDir: String?                     { didSet{onSet("dataDir")} }
	public var string = ""	                        { didSet{onSet("string")} }
	public var unzip  = true                        { didSet{onSet("unzip")} }
	
	// computed
	public var absoluteString: String { return string }

	//----------------------------------------------------------------------------
	// addAccessors
	public override func addAccessors() {
		super.addAccessors()
		addAccessor(name: "cacheDir",
		            get: { [unowned self] in self.cacheDir },
		            set: { [unowned self] in self.cacheDir = $0 })
		addAccessor(name: "cacheFile",
		            get: { [unowned self] in self.cacheFile },
		            set: { [unowned self] in self.cacheFile = $0 })
		addAccessor(name: "dataDir",
		            get: { [unowned self] in self.dataDir },
		            set: { [unowned self] in self.dataDir = $0 })
		addAccessor(name: "string",
		            get: { [unowned self] in self.string },
		            set: { [unowned self] in self.string = $0 })
		addAccessor(name: "unzip",
		            get: { [unowned self] in self.unzip },
		            set: { [unowned self] in self.unzip = $0 })
	}

	//----------------------------------------------------------------------------
	// getCacheFileURL
	public func getCacheFileURL(makeFolders: Bool) throws -> URL? {
		guard let cacheFile = self.cacheFile, !cacheFile.isEmpty else { return nil }
		let cacheFilePath = NSString(string: cacheFile).expandingTildeInPath
		let dirPath = NSString(string: cacheDir ?? "").expandingTildeInPath
		
		var cacheUrl = URL(string: cacheFilePath)
		if cacheUrl != nil {
			// if there is no scheme then its relative
			if cacheUrl!.scheme == nil {
				let dirUrl = URL(fileURLWithPath: dirPath, isDirectory: true)
				cacheUrl = URL(fileURLWithPath: cacheFilePath, relativeTo: dirUrl)
			}
			
			if makeFolders { try FileSystem.ensurePath(url: cacheUrl!) }
		}
		return cacheUrl
	}

	//----------------------------------------------------------------------------
	// getURL
	//  If locate is true then this will verify that the resource exists
	//
	public func getURL(locate: Bool = true) throws -> URL {
		let resourcePath = NSString(string: string).expandingTildeInPath

		guard var url = URL(string: resourcePath) else {
			writeLog("Failed to convert \(resourcePath) to URL")
			throw ModelError.conversationFailed(resourcePath)
		}
		
		// if there is no scheme then we assume its a local file
		if locate && url.scheme == nil {
			// check the data directory
			if let dataDir = self.dataDir {
				let dataDirPath = NSString(string: dataDir).expandingTildeInPath
				let dataDirUrl = URL(fileURLWithPath: dataDirPath, isDirectory: true)
				url = URL(fileURLWithPath: resourcePath, relativeTo: dataDirUrl)
				if FileSystem.fileExists(path: url.path).exists { return url }
			}
			
			if let storageLocation = try findStorageLocation() {
				url = storageLocation.appendingPathComponent(resourcePath)
				if FileSystem.fileExists(path: url.path).exists { return url }
			}
			
			// error
			writeLog("Unable to locate resource: \(string)")
		}
		
		return url
	}
	
	//----------------------------------------------------------------------------
	// ==
	public static func ==(lhs: Uri, rhs: Uri) -> Bool {
		return lhs.string == rhs.string
	}
}

