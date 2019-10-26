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

//==============================================================================
// Function
public final class Function : ModelElementContainerBase,
  XmlConvertible, JsonConvertible, Copyable, InitHelper {
	//-----------------------------------
	// initializers
	public required init() {
		super.init()
	}

	//-----------------------------------
	// load from URL
	public convenience init(log: Log?, contentsOf url: URL) throws {
		self.init()
		currentLog = log
		
		// set storage location of Function to support relative URLs
		storage = Uri(url: url.deletingLastPathComponent())
		
		// add elements to Function
		try update(fromXml: url)
	}

	//----------------------------------------------------------------------------
	// properties
	public var templateUri: Uri?                  { didSet{onSet("templateUri")} }
	
	//----------------------------------------------------------------------------
	// addAccessors
	public override func addAccessors() {
		super.addAccessors()
		addAccessor(name: "templateUri", lookup: .noLookup,
		            get: { [unowned self] in self.templateUri },
		            set: { [unowned self] in self.templateUri = $0 })
	}
	
	//----------------------------------------------------------------------------
	// setup
	public override func setup(taskGroup: TaskGroup) throws {
		// validate
		guard template == nil || templateUri == nil else {
			writeLog("\(namespaceName) template and templateUri" +
				" properties are mutually exclusive")
			throw ModelError.setupFailed
		}
		
		// optionally apply external Function properties
		if templateUri != nil { try applyExternalTemplate() }
		
		// setup base
		try super.setup(taskGroup: model.tasks)
		
		// connect
		try connectElements()
	}
	
	//----------------------------------------------------------------------------
	// applyExternalTemplate
	private func applyExternalTemplate() throws {
		guard let uri = templateUri else { return }
		if willLog(level: .diagnostic) {
			diagnostic("load external function template",
			           categories: .setup, trailing: "-", minCount: 60)
		}
		let element = try Function(log: currentLog, contentsOf: uri.getURL())
		try applyTemplate(templateElement: element)
	}
}
