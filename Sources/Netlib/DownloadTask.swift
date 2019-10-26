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
import Dispatch

final public class DownloadTask : ModelTaskBase, ModelTask {
	//----------------------------------------------------------------------------
	// properties
	var uri: Uri!
	let completionSemaphore = DispatchSemaphore(value: 0)
	private var session: URLSession!
	private var sessionTask: URLSessionDownloadTask?

	//----------------------------------------------------------------------------
	// begin with array handler
	func download(uri: Uri, group: DispatchGroup?,
	              handler: @escaping TaskArrayHandler) throws {
		self.uri = uri
		let url = try uri.getURL()
		
		session = URLSession(
			configuration: URLSessionConfiguration.default,
			delegate: DownloadDelegate(owner: self, handler: handler),
			delegateQueue: nil)
		
		// Initiate download
		DispatchQueue.global().async(group: group) { [unowned self] in
			self.writeLog("Downloading: \(uri.string)", level: .status)
			self.sessionTask = self.session.downloadTask(with: url)
			self.sessionTask!.resume()
			self.completionSemaphore.wait()
		}
	}

	//----------------------------------------------------------------------------
	// begin with URL handler
	func download(uri: Uri, group: DispatchGroup?,
	              handler: @escaping TaskUriHandler) throws {
		self.uri = uri
		let url = try uri.getURL()

		session = URLSession(
			configuration: URLSessionConfiguration.default,
			delegate: DownloadDelegate(owner: self, handler: handler),
			delegateQueue: nil)
		
		// Initiate download
		DispatchQueue.global().async(group: group) { [unowned self] in
			self.writeLog("Downloading: \(uri.string)", level: .status)
			self.sessionTask = self.session.downloadTask(with: url)
			self.sessionTask!.resume()
			self.completionSemaphore.wait()
		}
	}

	//----------------------------------------------------------------------------
	// cancel
	public override func cancel() {
		super.cancel()
		sessionTask?.cancel()
	}
}

//==============================================================================
// DownloadDelegate
//
final class DownloadDelegate : NSObject, URLSessionDownloadDelegate {
	// initializers
	init(owner: DownloadTask, handler: @escaping TaskArrayHandler) {
		self.owner = owner
		arrayHandler = handler
		super.init()
	}
	
	init(owner: DownloadTask, handler: @escaping TaskUriHandler) {
		self.owner = owner
		urlHandler = handler
		super.init()
	}

	//----------------------------------------------------------------------------
	// properties
	unowned let owner: DownloadTask
	var arrayHandler: TaskArrayHandler!
	var urlHandler: TaskUriHandler!
	
	//----------------------------------------------------------------------------
	// didFinishDownloadingTo
	func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
	                didFinishDownloadingTo location: URL) {
		// now matter how we exit, make sure to signal at the end that we are done
		defer { owner.completionSemaphore.signal() }

		do {
			// cache?
			if let cacheURL = try owner.uri.getCacheFileURL(makeFolders: true) {
				try FileSystem.copyItem(at: location, to: cacheURL)
			}

			// notify handler
			if let urlHandler = self.urlHandler {
				urlHandler(location)

			} else {
				// load the data and call the handler
				var data = try [UInt8](contentsOf: location)
				if owner.uri.string.contains(".gz") {
					owner.diagnostic("unzipping: \(location.path)",
						categories: .download, indent: 1)
					data = try unzip(data: data)
				}
				arrayHandler(data)
			}

		} catch {
			owner.writeLog(String(describing: error))
			owner.cancel()
		}
	}
}



