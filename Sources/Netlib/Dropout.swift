//******************************************************************************
//  Created by Edward Connell on 5/26/17
//  Copyright © 2016 Connell Research. All rights reserved.
//
//  https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf
//
final public class Dropout : ComputableFilterBase, DropoutProperties
{
	//----------------------------------------------------------------------------
	// properties
	public var drop = 0.5                               { didSet{onSet("drop")} }
	public var seed: UInt?                              { didSet{onSet("seed")} }

	//----------------------------------------------------------------------------
	// addAccessors
	public override func addAccessors() {
		super.addAccessors()
		addAccessor(name: "drop",
		            get: { [unowned self] in self.drop },
		            set: { [unowned self] in self.drop = $0 })
		addAccessor(name: "seed" ,
		            get: { [unowned self] in self.seed } ,
		            set: { [unowned self] in self.seed = $0 })
	}

	//----------------------------------------------------------------------------
	// setup
	public override func setup(taskGroup: TaskGroup) throws {
		try super.setup(taskGroup: taskGroup)
		guard 0.0...1.0 ~= drop else {
			writeLog("\(namePath) drop \(drop) is out of range 0.0...1.0")
			throw ModelError.setupFailed
		}
	}
}

//==============================================================================
//
public protocol DropoutProperties : ComputableFilterProperties
{
	var drop: Double { get }
	var seed: UInt? { get }
}
