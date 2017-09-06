//******************************************************************************
//  Created by Edward Connell on 4/11/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
final public class LrnCrossChannel :
	ComputableFilterBase, LrnCrossChannelProperties, InitHelper {
	//----------------------------------------------------------------------------
	// properties
	public var alpha = 1e-4                                { didSet{onSet("alpha")} }
	public var beta = 0.75                                 { didSet{onSet("beta")} }
	public var k = 2.0                                     { didSet{onSet("k")} }
	public var mode = LrnCrossChannelMode.crossChannelDim1 { didSet{onSet("mode")} }
	public var windowSize: Int = 5                         { didSet{onSet("windowSize")} }

	//----------------------------------------------------------------------------
	// addAccessors
	public override func addAccessors() {
		super.addAccessors()
		addAccessor(name: "alpha",
		            get: { [unowned self] in self.alpha },
		            set: { [unowned self] in self.alpha = $0 })
		addAccessor(name: "beta",
		            get: { [unowned self] in self.beta },
		            set: { [unowned self] in self.beta = $0 })
		addAccessor(name: "k",
		            get: { [unowned self] in self.k },
		            set: { [unowned self] in self.k = $0 })
		addAccessor(name: "mode",
		            get: { [unowned self] in self.mode },
		            set: { [unowned self] in self.mode = $0 })
		addAccessor(name: "windowSize",
		            get: { [unowned self] in self.windowSize },
		            set: { [unowned self] in self.windowSize = $0 })
	}
}

//==============================================================================
//
public protocol LrnCrossChannelProperties : ComputableFilterProperties {
	var alpha: Double { get }
	var beta: Double { get }
	var k: Double { get }
	var mode: LrnCrossChannelMode { get }
	var windowSize: Int { get }
}

// LrnCrossChannelMode
public enum LrnCrossChannelMode : String, EnumerableType {
	case crossChannelDim1
}



