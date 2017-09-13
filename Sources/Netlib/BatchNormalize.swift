//******************************************************************************
//  Created by Edward Connell on 10/19/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
//  https://arxiv.org/pdf/1502.03167.pdf
//
final public class BatchNormalize :
	ComputableFilterBase, BatchNormalizeProperties, InitHelper {
	//----------------------------------------------------------------------------
	// properties
	public var epsilon = 1e-5                    { didSet{onSet("epsilon")} }
	public var momentum = 0.9                    { didSet{onSet("momentum")} }
	public var mode = BatchNormalizeMode.auto    { didSet{onSet("mode")} }
	public var running_mean: DataView?
	public var running_var: DataView?

	//----------------------------------------------------------------------------
	// addAccessors
	public override func addAccessors() {
		super.addAccessors()
		addAccessor(name: "epsilon",
		            get: { [unowned self] in self.epsilon },
		            set: { [unowned self] in self.epsilon = $0 })
		addAccessor(name: "momentum",
		            get: { [unowned self] in self.momentum },
		            set: { [unowned self] in self.momentum = $0 })
		addAccessor(name: "mode",
		            get: { [unowned self] in self.mode },
		            set: { [unowned self] in self.mode = $0 })
	}

	//----------------------------------------------------------------------------
	// copy
	public override func copy(from other: Properties) {
		super.copy(from: other)
		let other = other as! BatchNormalize
		running_mean = other.running_mean
		running_var  = other.running_var
	}
}

//==============================================================================
//
public protocol BatchNormalizeProperties : ComputableFilterProperties {
	var epsilon: Double { get }
	var momentum: Double { get }
	var mode: BatchNormalizeMode { get }
	var running_mean: DataView? { get set }
	var running_var: DataView? { get set }
}

// BatchNormalizeMode
public enum BatchNormalizeMode : String, EnumerableType {
	case auto, perActivation, spatial
}
