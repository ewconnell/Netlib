//******************************************************************************
//  Created by Edward Connell on 5/9/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
//	inspired by: http://blog.scottlogic.com/2015/02/05/swift-events.html
//
import Foundation
import Dispatch

public class DisposableStub : Disposable {
	public init() {}
	public func dispose() { }
}
public protocol Invocable: class { func invoke(data: Any) }
public protocol Disposable { func dispose() }

public typealias EventFunction<T> = (T) -> ()
public typealias EventClosure<T> = (T) -> Void

//------------------------------------------------------------------------------
// Event
public class Event<T> {
	// initializers
	public init() {}

	// properties
	public var eventHandlers = [Invocable]()
	
	// functions
	public func raise(data: T) {
		for handler in eventHandlers { handler.invoke(data: data) }
	}

	// adds a closure type handler that allows the caller to control disposal
	public func addHandler(queue: DispatchQueue? = nil,
	                       handler: @escaping EventClosure<T>) -> Disposable {
		let disp = ClosureWrapper(queue: queue, handler: handler, event: self)
		eventHandlers.append(disp)
		return disp
	}

	// adds a class member func as handler
	//
	// Example: where self is HandlerOwnerClass
	//	event.addHandler(target: self, handler: HandlerOwnerClass.myChangedHandler)
	//
	public func addHandler<U: AnyObject>(target: U,	queue: DispatchQueue? = nil,
	                       handler: @escaping (U) -> EventFunction<T>) -> Disposable {
		eventHandlers.append(FunctionWrapper(target: target, queue: queue,
																				 handler: handler, event: self))
		return eventHandlers.last as! Disposable
	}
}

//------------------------------------------------------------------------------
// ClosureWrapper
private class ClosureWrapper<T> : Invocable, Disposable {
	// initializers
	init(queue: DispatchQueue?,
	     handler: @escaping EventClosure<T>, event: Event<T>) {
		self.queue = queue
		self.handler = handler
		self.event = event;
	}

	deinit { dispose() }

	// properties
	var handler: EventClosure<T>
	weak var queue: DispatchQueue?
	weak var event: Event<T>?

	// functions
	func invoke(data: Any) -> () {
		if let queue = self.queue {
			queue.async() { [unowned self] in self.handler(data as! T) }
		} else {
			handler(data as! T)
		}
	}
	
	func dispose() {
		if let event = self.event {
			event.eventHandlers =	event.eventHandlers.filter { $0 !== self }
		}
	}
}

//------------------------------------------------------------------------------
// FunctionWrapper
private class FunctionWrapper<T: AnyObject, U> : Invocable, Disposable {
	// initializers
	init(target: T?, queue: DispatchQueue?,
	     handler: @escaping (T) -> EventFunction<U>, event: Event<U>) {
		self.queue = queue
		self.target = target
		self.handler = handler
		self.event = event;
	}
	deinit { dispose() }
	
	// properties
	let handler: (T) -> (U) -> ()
	weak var queue: DispatchQueue?
	weak var target: T?
	weak var event: Event<U>?
	
	// functions
	func invoke( data: Any) -> () {
		if let target = self.target {
			if let queue = self.queue {
				queue.async() { [unowned self] in self.handler(target)(data as! U) }
			} else {
				handler(target)(data as! U)
			}
		}
	}
	
	func dispose() {
		if let event = self.event {
			event.eventHandlers =	event.eventHandlers.filter { $0 !== self }
		}
	}
}

