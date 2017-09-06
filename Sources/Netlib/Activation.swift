//******************************************************************************
//  Created by Edward Connell on 5/25/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
public final class Activation : ComputableFilterBase, ActivationProperties, InitHelper {
	//----------------------------------------------------------------------------
	// properties
	public var mode = ActivationMode.relu         { didSet{onSet("mode")} }
	public var nan = NanPropagation.propagate	    { didSet{onSet("nan")} }
	public var reluCeiling = 0.0	                { didSet{onSet("reluCeiling")} }

	//----------------------------------------------------------------------------
	// addAccessors
	public override func addAccessors() {
		super.addAccessors()
		addAccessor(name: "mode",
		            get: { [unowned self] in self.mode },
		            set: { [unowned self] in self.mode = $0 })
		addAccessor(name: "nan",
		            get: { [unowned self] in self.nan },
		            set: { [unowned self] in self.nan = $0 })
		addAccessor(name: "reluCeiling",
		            get: { [unowned self] in self.reluCeiling },
		            set: { [unowned self] in self.reluCeiling = $0 })
	}
}

//==============================================================================
//
public protocol ActivationProperties : ComputableFilterProperties {
	var mode: ActivationMode { get }
	var nan: NanPropagation { get }
	var reluCeiling: Double { get }
}

//--------------------------------------
// ActivationMode
public enum ActivationMode : String, EnumerableType {
	case sigmoid, relu, tanh, clippedRelu
}
