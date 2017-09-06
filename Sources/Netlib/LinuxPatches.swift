//******************************************************************************
//  Created by Edward Connell on 10/26/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
import Foundation

#if os(Linux)
	import Glibc
#endif


// getDataDirectory
public func getDataDirectory() -> String {
	var path = "~/Documents/data"
	#if os(Linux)
		if let homeDir = ProcessInfo.processInfo.environment["HOME"] {
			path = URL(fileURLWithPath: homeDir, isDirectory: true)
				.appendingPathComponent("Documents/data").path
		}
	#else
		path = FileManager.default.urls(
			for: .documentDirectory, in: .userDomainMask)[0]
			.appendingPathComponent("data").path
	#endif
	return path
}

// getBundle
public func getBundle(for anyClass: AnyClass) -> Bundle {
	#if os(Linux)
	return Bundle.main
	#else
	return Bundle(for: anyClass)
	#endif
}

// random_uniform
public func random_uniform(range: Int) -> Int {
	guard range > 0 else { return 0 }
	#if os(Linux)
	  return Int(random()) % range
	#else
		return Int(arc4random_uniform(UInt32(range)))
	#endif
}

class FileSystem {
	class func fileExists(path: String) -> (exists: Bool, isDir: Bool) {
		var temp: ObjCBool = false
		let exists = FileManager.default.fileExists(atPath: path, isDirectory: &temp)
#if os(Linux)
		return (exists, temp as Bool)
#else
		return (exists, temp.boolValue)
#endif
	}

	class func ensurePath(url: URL) throws {
		// create intermediate directories
		let folderPath = url.deletingLastPathComponent().path
		if !FileSystem.fileExists(path: folderPath).exists {
			// create data folder
			try FileManager.default.createDirectory(
				atPath: folderPath, withIntermediateDirectories: true, attributes: nil)
		}
	}

	class func copyItem(at source: URL, to output: URL) throws {
		let data = try Data(contentsOf: source)
		try data.write(to: output, options: .atomic)
	}
}
