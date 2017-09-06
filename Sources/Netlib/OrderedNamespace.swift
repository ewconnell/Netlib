//******************************************************************************
//  Created by Edward Connell on 4/19/17
//  Copyright © 2017 Connell Research. All rights reserved.
//
import Foundation

//==============================================================================
// OrderedNamespace
//  This collection maintains a list of Ts that is both ordered
// and indexed by name.
public final class OrderedNamespace<T: ModelObject> : Sequence {
	//----------------------------------------------------------------------------
	// Properties
	public private(set) var itemsIndex = [String : Int]()
	public private(set) var items = [T]()
	
	// events
	public var changed = Event<OrderedNamespace>()

	//----------------------------------------------------------------------------
	// makeIterator
	public func makeIterator() -> OrderedNamespaceIterator<T> {
		return OrderedNamespaceIterator(collection: self)
	}

	//------------------------------------
	// count
	public var count: Int { return items.count }
	public var isEmpty: Bool { return items.count == 0 }

	//------------------------------------
	// copy
	public func copy() -> OrderedNamespace {
		let ns = OrderedNamespace()
		var copied = [T]()
		for item in items { copied.append(item.copy(type: T.self)) }
		ns.items = copied
//		ns.items = items.map { $0.copy(type: T.self) }
		ns.itemsIndex = itemsIndex
		return ns
	}

	//------------------------------------
	// append
	@discardableResult
	public func append<U: ModelObject>(_ item: U) -> U {
		let itemIndex = items.count
		let storedItem = item as! T
		items.append(storedItem)
		addItemToIndex(item: storedItem, itemIndex: itemIndex)
		assert(items.count == itemsIndex.count)
		
		// monitor changes for the items "name" property
		_ = item.properties["name"]!.changed.addHandler(
			target: self, handler: OrderedNamespace.onItemNameChanged)
		changed.raise(data: self)
		return item
	}

	//------------------------------------
	// append
	public func append(_ modelObjects: [ModelObject]) {
		for object in modelObjects {
			let itemIndex = items.count
			let storedItem = object as! T
			items.append(storedItem)
			addItemToIndex(item: storedItem, itemIndex: itemIndex)
			assert(items.count == itemsIndex.count)

			// monitor changes for the items "name" property
			_ = object.properties["name"]!.changed.addHandler(
				target: self, handler: OrderedNamespace.onItemNameChanged)
		}

		changed.raise(data: self)
	}

	//------------------------------------
	// addItemToIndex
	private func addItemToIndex(item: T, itemIndex: Int) {
		item.namespaceName = item.name ?? "\(item.typeName.lowercased())(\(itemIndex))"
		if itemsIndex[item.namespaceName] != nil {
			item.writeLog("\(item.namePath): 'name' must be unique or nil")
			item.namespaceName = "\(item.typeName.lowercased())(\(itemIndex))"
		}
		itemsIndex[item.namespaceName] = itemIndex
	}

	//------------------------------------
	// onItemNameChanged
	private func onItemNameChanged(sender: Properties) {
		for (name, index) in itemsIndex {
			if items[index] === sender {
				itemsIndex.removeValue(forKey: name)
				addItemToIndex(item: items[index], itemIndex: index)
			}
		}
	}
	
	//------------------------------------
	// insert
	public func insert(_ item: T, at: Int) {
		// TODO
		fatalError("not implemented yet")
//		changed.raise(data: self)
	}

	//------------------------------------
	// insert
	public var last: T? {
		return items.last
	}
	
	//------------------------------------
	// remove
	public func remove(item: T) {
		for i in 0..<items.count {
			if items[i] === item {
				remove(at: i)
				return
			}
		}
	}

	//------------------------------------
	// remove
	@discardableResult
	public func remove(name: String) -> T? {
		guard let index = itemsIndex[name] else { return nil }
		return remove(at: index)
	}

	//------------------------------------
	// remove
	@discardableResult
	public func remove(at index: Int) -> T {
		let item = items.remove(at: index)
		itemsIndex.removeValue(forKey: item.namespaceName)
		changed.raise(data: self)
		return item
	}

	//------------------------------------
	// removeAll
	public func removeAll() {
		items.removeAll()
		itemsIndex.removeAll()
	}
	
	//------------------------------------
	// subscript(index: Int)
	public subscript(index: Int) -> T {
		get { return items[index] }
	}

	//------------------------------------
	// subscript(index: String)
	public subscript(index: String) -> T? {
		get {
			guard let i = itemsIndex[index] else { return nil }
			return items[i]
		}
	}
}

//==============================================================================
// OrderedNamespaceIterator
public struct OrderedNamespaceIterator<T: ModelObject>: IteratorProtocol {
	public init(collection: OrderedNamespace<T>) {
		self.collection = collection
	}

	private var index = -1
	private let collection: OrderedNamespace<T>

	public mutating func next() -> T? {
		index += 1
		return index < collection.items.count ? collection[index] : nil
	}
}
