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

typealias StringRange = Range<String.Index>

//==============================================================================
// XmlDocument
public class XmlDocument : Logging {
	// initializers
	public init(log: Log?, rootElement: XmlElement) {
		self.rootElement = rootElement
		currentLog = log
	}
	
	public init(log: Log?, string: String) throws {
		currentLog = log
		if let next = try parseHeader(xml: string) {
			_ = try parseElements(parent: nil, xml: string,
				                    range: next..<string.endIndex)
		}
	}
	
	public convenience init(log: Log?, contentsOf url: URL) throws {
		#if os(Linux)
			// TODO: String(contentsOf is not implemented on Linux yet
			let data = try Data(contentsOf: url)
			let string = String(data: data, encoding: .utf8)!
		#else
			let string = try String(contentsOf: url)
		#endif
		try self.init(log: log, string: string)
	}

	public convenience init(log: Log?, data: [UInt8]) throws {
		if let string = String(bytes: data, encoding: .utf8) {
			try self.init(log: log, string: string)
		} else {
			throw ModelError.error("xml string conversion failed")
		}
	}

	//----------------------------------------------------------------------------
	// properties
	public var version = "1.0"
	public var encoding: String.Encoding = .utf8
	public var rootElement: XmlElement?

	// logging
	public var logLevel = LogLevel.error
	public var nestingLevel = 0
	public weak var currentLog: Log?

	private lazy var attrTrimChars: CharacterSet = {
		var set = CharacterSet.whitespacesAndNewlines
		set.insert("=")
		set.insert("<")
		set.insert(">")
		return set
	}()
	
	private lazy var nameBreakChars: CharacterSet = {
		var set = CharacterSet.whitespacesAndNewlines
		set.insert("/")
		return set
	}()
	
	//----------------------------------------------------------------------------
	// xmlString
	public func xmlString(format: Bool = false, tabSize: Int? = nil) -> String {
		var string = getHeader(format: format, tabSize: tabSize)
		if let root = rootElement {
			append(element: root, to: &string, format: format, tabSize: tabSize)
		}
		return string
	}

	//-------------------------------------
	// getHeader
	private func getHeader(format: Bool, tabSize: Int?) -> String {
		let newLine = format ? "\n" : ""
		// TODO: add more??
		let encodings: [String.Encoding : String] = [.utf8 : "utf-8"]
		return "<?xml version=\"\(version)\" encoding=\"\(encodings[encoding]!)\"?>\(newLine)"
	}
	

	//-------------------------------------
	// append(element
	private func append(element: XmlElement, to string: inout String,
	                    format: Bool, tabSize: Int?)
	{
		var beginLine = string.endIndex
		let newLine = format ? "\n" : ""
		let indent = tabFiller(count: element.level, tabSize: tabSize)
		string.append("\(indent)<")
//		if !element.namespace.isEmpty {	string.append("\(element.namespace):") }
		let beginElement = string.endIndex
		string.append(element.name)

		if format {
			let beginAttrDist = string.distance(from: beginElement, to: string.endIndex)
			let attrIndent = indent + String(repeating: " ", count: beginAttrDist)
			for attr in element.attributes {
				let attrString = " \(attr.name)=\"\(attr.stringValue)\""
				let currentCol = string.distance(from: beginLine, to: string.endIndex)
				if currentCol + attrString.count > 80 {
					string.append(newLine)
					beginLine = string.endIndex
					string.append(attrIndent)
				}
				string.append(attrString)
			}
		} else {
			for attr in element.attributes {
				string.append(" \(attr.name)=\"\(attr.stringValue)\"")
			}
		}
		
		if element.children.isEmpty {
			string.append("/>\(newLine)")
		} else {
			string.append(">\(newLine)")
			for child in element.children {
				append(element: child, to: &string, format: format, tabSize: tabSize)
			}
			string.append("\(indent)</\(element.name)>\(newLine)")
		}
	}

	//-------------------------------------
	// tabFiller
	private func tabFiller(count: Int, tabSize: Int?) -> String {
		if let tabSize = tabSize {
			return String(repeating: " ", count: count * tabSize)
		} else {
			return String(repeating: "\t", count: count)
		}
	}

	//----------------------------------------------------------------------------
	// parseHeader
	private func parseHeader(xml: String)	throws -> String.Index?	{
		guard xml.range(of: "<?xml") != nil else { return xml.startIndex }
		guard let headerClosing = xml.range(of: "?>") else {
			writeLog("document heading must be closed with '?>'")
			throw XmlConvertibleError.parsingError
		}
		return headerClosing.upperBound
	}

	//----------------------------------------------------------------------------
	// parseElements
	private func parseElements(parent: XmlElement?, xml: String,
	                           range searchRange: StringRange)
		throws -> String.Index?
	{
		// find a valid element header definition
		guard !searchRange.isEmpty,
			let headerOpening = xml.range(of: "<", range: searchRange) else {
			return nil
		}
		
		// we opened an element, so there must be a next character
		guard let nextChar = getNextChar(xml: xml, index: headerOpening.lowerBound) else {
			writeLog("element is not closed: \(xml)")
			throw XmlConvertibleError.parsingError
		}
		
		// return if this is an element body closing
		if nextChar == "/" { return nil }
		
		// skip comment
		if nextChar == "!" {
			guard let commentClosing = xml.range(of: "-->", range: searchRange) else {
				let comment = xml[headerOpening.lowerBound..<searchRange.upperBound]
				writeLog("comment must be closed \(comment)")
				throw XmlConvertibleError.parsingError
			}
			return commentClosing.upperBound == searchRange.upperBound ?
				nil : commentClosing.upperBound
		}

		// get closing
		guard let headerClosing = xml.range(of: ">", range: searchRange) else {
			writeLog("element is not closed: \(xml)")
			throw XmlConvertibleError.parsingError
		}
		guard headerClosing.lowerBound > headerOpening.upperBound else {
			// create a message window
			let lower = xml.index(headerClosing.lowerBound,
				offsetBy: max(-40, -xml.distance(from: xml.startIndex, to: headerClosing.upperBound)))

			let upper = xml.index(headerClosing.upperBound,
				offsetBy: min(40, xml.distance(from: headerClosing.upperBound, to: xml.endIndex)))

			let range: Range<String.Index> = (lower..<upper).clamped(to: xml.startIndex..<xml.endIndex)
			let message = "improperly nested elements: (\(String(xml[range])))"
			writeLog(message)
			throw XmlConvertibleError.parsingError
		}
		let headerContentRange: Range<String.Index> = headerOpening.upperBound..<headerClosing.lowerBound
		
		// get element name range
		var nameRange = headerContentRange
		if let nameBreak = xml.rangeOfCharacter(from: nameBreakChars,
		                                        range: headerContentRange) {
			nameRange = nameRange.lowerBound..<nameBreak.lowerBound
		}
		
		if nameRange.isEmpty {
			writeLog("missing xml element name")
			throw XmlConvertibleError.parsingError
		}
		
		// add child
		let element = XmlElement(name: String(xml[nameRange]))
		if let parent = parent {
			element.level = parent.level + 1
			parent.add(child: element)
		} else {
			rootElement = element
		}
		
		// get attributes
		let attributeRange: Range<String.Index> = nameRange.upperBound..<headerClosing.lowerBound
		try parseAttributes(elt: element, xml: xml, range: attributeRange)
		
		// check if has body
		let hasBody = String(xml[headerContentRange])
			.trimmingCharacters(in: .whitespacesAndNewlines).last != "/"

		// recursively dive down to parse the body
		let eltClosing: StringRange
		if hasBody {
			var range: Range<String.Index> = headerClosing.upperBound..<searchRange.upperBound
			while let next = try parseElements(parent: element, xml: xml, range: range) {
				range = next..<searchRange.upperBound
			}
			
			// find the end
			guard let closing = xml.range(of: "</\(element.name)", range: range),
				let end = xml.range(of: ">", range: closing.upperBound..<range.upperBound) else {
				writeLog("element is not closed: \(element.name)")
				throw XmlConvertibleError.parsingError
			}
			
			eltClosing = closing.lowerBound..<end.upperBound
		} else {
			eltClosing = headerClosing
		}
		
		// next
		let next: String.Index? = eltClosing.upperBound == searchRange.upperBound ?
			nil : eltClosing.upperBound

		// return
		return next
	}
	
	//----------------------------------------------------------------------------
	// getNextIndex
	private func getNextIndex(xml: String, index: String.Index) -> String.Index? {
		let next = xml.index(after: index)
		return next == xml.endIndex ? nil : next
	}

	//----------------------------------------------------------------------------
	// getNextChar
	private func getNextChar(xml: String, index: String.Index) -> Character? {
		if let nextIndex = getNextIndex(xml: xml, index: index) {
			return xml[nextIndex]
		} else {
			return nil
		}
	}
	
	//----------------------------------------------------------------------------
	// parseAttributes
	private func parseAttributes(elt: XmlElement, xml: String, range: StringRange) throws {
		guard !range.isEmpty else { return }
		var attrs = String(xml[range]).components(separatedBy: "\"")
		
		for i in stride(from: 0, to: attrs.count - 1, by: 2) {
			let name = attrs[i].trimmingCharacters(in: attrTrimChars)
			elt.add(attribute: XmlAttribute(withName: name, stringValue: attrs[i+1]))
		}
	}
}

//==============================================================================
// XmlAttribute
public class XmlAttribute {
	// initializers
	public init(withName: String, stringValue: String) {
		name = withName
		self.stringValue = stringValue
	}
	
	//----------------------------------------------------------------------------
	// properties
	public var name: String
	public var stringValue: String
}

//==============================================================================
// XmlElement
public class XmlElement {
	// initializers
	public init() { }
	
	public init(name: String, level: Int = 0, kind: XmlElementKind = .element) {
		self.name = name
		self.kind = kind
		self.level = level
	}
	
	//----------------------------------------------------------------------------
	// properties
	public var name = ""
	public var kind = XmlElementKind.element
	public var attributes = [XmlAttribute]()
	public var children = [XmlElement]()
	public var namespace = ""
	public var level = 0
	
	public func add(child: XmlElement) { children.append(child)	}
	public func add(namespace: String) { self.namespace = namespace	}
	public func add(attribute: XmlAttribute) {
		attributes.append(attribute)
	}
}

public enum XmlElementKind {
	case header, element, comment
}


//----------------------------------------------------------------------------
// reIndent
public func reIndent(text: String, tabSize: Int) -> String {
	let lines = text.components(separatedBy: "\n")
	var result = ""
	for line in lines {
		if let firstChar = line.index(where: { $0 != " "}) {
			let count: Int = line.distance(from: line.startIndex,
			                                          to: firstChar) / 4 * tabSize
			result += String(repeating: " ", count: count) +
				String(line[firstChar...]) + "\n"
		} else {
			result += line + "\n"
		}
	}
	return result
}
