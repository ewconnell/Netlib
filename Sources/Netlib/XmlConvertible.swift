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

let NetlibNamespace = "http://connellresearch.com/Netlib"

//==============================================================================
// XmlConvertible
public protocol XmlConvertible {
	func asXml(after modelVersion: Int, format: Bool, tabSize: Int) -> String
	func writeXml(to stream: OutputStream, after modelVersion: Int,
	              format: Bool, tabSize: Int) throws -> Int
	
	func update(fromXml url: URL) throws
	func update(fromXml string: String) throws
	func update(fromXml stream: InputStream) throws

	func updateSet(fromXml url: URL) throws -> AnySet
}

public enum XmlConvertibleError: Error {
	case parsingError
	case stringEncodingError
	case streamEncodingError
}

//==============================================================================
// XmlConvertible extension
extension XmlConvertible where Self: ModelObject {
	//----------------------------------------------------------------------------
	// asXml
	public func asXml(after modelVersion: Int, format: Bool, tabSize: Int) -> String {
		return CreateXmlDocument(after: modelVersion, from: self, log: currentLog)
			.xmlString(format: format, tabSize: tabSize)
	}

	//----------------------------------------------------------------------------
	// writeXml
	public func writeXml(to stream: OutputStream, after modelVersion: Int,
	                     format: Bool, tabSize: Int) throws -> Int {
		let xmlString =
			CreateXmlDocument(after: modelVersion, from: self, log: currentLog)
				.xmlString(format: format, tabSize: tabSize)

		return stream.write(xmlString, maxLength: xmlString.count)
	}

	//----------------------------------------------------------------------------
	// update(fromXml string:
	public func update(fromXml string: String) throws {
		try updateAny(with: try XmlToAny(doc: XmlDocument(log: currentLog,
		                                                  string: string)))
	}

	//----------------------------------------------------------------------------
	// updateSet(fromXml url:
	public func updateSet(fromXml url: URL) throws -> AnySet {
		return try XmlToAny(doc: XmlDocument(log: currentLog, contentsOf: url))
	}

	//----------------------------------------------------------------------------
	// update(fromXml url:
	public func update(fromXml url: URL) throws {
		try updateAny(with: updateSet(fromXml: url))
	}
	
	//----------------------------------------------------------------------------
	// update(fromXml stream:
	public func update(fromXml stream: InputStream) throws {
		assertionFailure("not implemented")
	}

	//----------------------------------------------------------------------------
	// XmlToAny
	public func XmlToAny(doc: XmlDocument) throws -> AnySet {
		guard let root = doc.rootElement else { return AnySet() }
		return try GetObject(elt: root)
	}

	//----------------------------------------------------------------------------
	// GetObject
	private func GetObject(elt: XmlElement) throws -> AnySet {
		var result = AnySet()
		let typeName = elt.name
		result[TypeNameKey] = typeName

		// create an instance to access it's type information
		let object = try Create(typeName: typeName)

		// get object attributes
		for attr in elt.attributes {
			if let prop = object.properties[attr.name] {
				switch prop.propertyType {
				case .attribute: result[attr.name] = attr.stringValue

					// embedded xml object
				case .object:
					// convert xml syntax for outer most embedded object
					var level = -1
					let xmlString = String(attr.stringValue.map({
						(char: Character) -> Character in
						switch char {
						case "{":
							level += 1
							return level == 0 ? "<" : char

						case "}":
							defer {
								level -= 1
							}
							return level == 0 ? ">" : char

						case "'" where level == 0: return "\""
						default: return char
						}
					})).trim()

					// make sure the embedded has an opening >
					if xmlString.first != "<" {
						writeLog("Embedded xml object expected. Found: \(xmlString)")
						throw XmlConvertibleError.parsingError
					}

					do {
						let embedded = try XmlDocument(log: currentLog, string: xmlString)
						result[attr.name] = try XmlToAny(doc: embedded)
					} catch {
						writeLog("Error parsing embedded content \"\(xmlString)\" - \(error)")
						throw XmlConvertibleError.parsingError
					}
					break
				case .collection:
					writeLog("Error parsing collection used as attribute \(prop.name)")
					throw XmlConvertibleError.parsingError
				}

			} else {
				writeLog("Unrecognized attribute ignored: \(elt.name).\(attr.name)")
			}
		}

		// get embedded attributes
		for attr in elt.children where attr.kind == .element {
			if let prop = object.properties[attr.name] {
				switch prop.propertyType {
				case .object:
					var found = false
					for child in attr.children {
						if found {
							writeLog("Attribute is not a array - ignored: " +
								"\(elt.name).\(attr.name)")
							break
						} else {
							let obj = try GetObject(elt: child)
							result[attr.name] = obj
							found = true
						}
					}

				case .collection:
					var array = [AnySet]()
					for child in attr.children {
						array.append(try GetObject(elt: child))
					}
					result[attr.name] = array

				case .attribute:
					writeLog("Simple attributes should not be embedded - ignored: " +
						"\(elt.name).\(attr.name)")
				}
			} else {
				writeLog("Unrecognized attribute ignored: \(elt.name).\(attr.name)")
			}
		}
		return result
	}

	//============================================================================
	// CreateXmlDocument
	private func CreateXmlDocument(after modelVersion: Int,
	                               from source: Properties,
	                               log: Log?) -> XmlDocument {
		// when Self is an object, a dictionary of Any is returned
		var selected = source.selectAny(after: modelVersion,
		                                include: [.types]) ?? AnySet()

		selected.removeValue(forKey: VersionKey)

		let rootElement = selected.isEmpty ? XmlElement() :
			AddElement(object: selected, level: 0)

		rootElement.add(namespace: NetlibNamespace)
		let doc = XmlDocument(log: log, rootElement: rootElement)
		return doc
	}

	//----------------------------------------------------------------------------
	private func AddElement(object: AnySet, level: Int) -> XmlElement {
		// add type
		let typeName = object[TypeNameKey]! as! String
		let elt = XmlElement(name: typeName, level: level)

		for (key, value) in object
			where key != TypeNameKey && !(value is NSNull) {
				
			switch value {
			case let child as AnySet:
				let object = XmlElement(name: key, level: level + 1)
				elt.add(child: object)
				object.add(child: AddElement(object: child, level: level + 2))

			case let arrayValues as [AnySet]:
				let array = XmlElement(name: key, level: level + 1)
				elt.add(child: array)
				for item in arrayValues {
					array.add(child: AddElement(object: item, level: level + 2))
				}

			case let value as [Int]:
				var strval = String(describing: value)
				let startIndex = strval.index(after: strval.startIndex)
				let endIndex = strval.index(before: strval.endIndex)
				strval = String(strval[startIndex..<endIndex])
				elt.add(attribute: XmlAttribute(withName: key, stringValue: strval))

			default:
				elt.add(attribute: XmlAttribute(withName: key, stringValue: "\(value)"))
			}
		}
		return elt
	}
}




