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
#if swift(>=4.1)
		return (exists, temp.boolValue)
#else
		return (exists, temp as Bool)
#endif
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
