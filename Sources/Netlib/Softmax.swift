//******************************************************************************
//  Created by Edward Connell on 4/11/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
final public class Softmax : ComputableFilterBase, SoftmaxProperties, InitHelper {
	//----------------------------------------------------------------------------
	// properties
	public var algorithm = SoftmaxAlgorithm.accurate { didSet{onSet("algorithm")} }
	public var mode = SoftmaxMode.channel            { didSet{onSet("mode")} }
	public var outputType = SoftmaxOutput.labels     { didSet{onSet("outputType")} }

	//----------------------------------------------------------------------------
	// addAccessors
	public override func addAccessors() {
		super.addAccessors()
		addAccessor(name: "algorithm",
		            get: { [unowned self] in self.algorithm },
		            set: { [unowned self] in self.algorithm = $0 })
		addAccessor(name: "mode",
		            get: { [unowned self] in self.mode },
		            set: { [unowned self] in self.mode = $0 })
		addAccessor(name: "outputType",
		            get: { [unowned self] in self.outputType },
		            set: { [unowned self] in self.outputType = $0 })
	}
}

//==============================================================================

public protocol SoftmaxProperties : ComputableFilterProperties {
	var algorithm: SoftmaxAlgorithm { get }
	var mode: SoftmaxMode { get }
	var outputType: SoftmaxOutput { get }
}

public enum SoftmaxAlgorithm : String, EnumerableType {
	case accurate, fast, log
}

public enum SoftmaxMode : String, EnumerableType {
	case channel, instance
}

public enum SoftmaxOutput : String, EnumerableType {
	case labels, probabilities
}

