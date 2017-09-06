//******************************************************************************
//  Created by Edward Connell on 4/11/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
import Foundation
#if os(Linux)
	import Glibc
#endif


final public class LearnedParameter : ModelObjectBase, InitHelper {
	//----------------------------------------------------------------------------
	// properties
	public var fillMethod = DataFill.zero	         { didSet{onSet("fillMethod")} }
	public var fillIndexStartAt = 0	               { didSet{onSet("fillIndexStartAt")}}
	public var fillMean = 0.0	                     { didSet{onSet("fillMean")} }
	public var fillStd = 1.0	                     { didSet{onSet("fillStd")} }
	public var fillRange = [-1.0, 1.0]	           { didSet{onSet("fillRange")} }
	public var fillValue = 0.0	                   { didSet{onSet("fillValue")} }
	public var fillVarianceNorm: FillVarianceNorm? { didSet{onSet("fillVarianceNorm")}}
	public var learningRateScale = 1.0             { didSet{onSet("learningRateScale")}}
	public var seed: UInt?                         { didSet{onSet("seed")} }
	public var uri: Uri?       	                   { didSet{onSet("uri")} }
	public var uriDataIsColMajor = false           { didSet{onSet("uriDataIsColMajor")}}
	public var uriDataType = DataType.real32F      { didSet{onSet("uriDataType")} }
	public var uriDataExtent: [Int]?	             { didSet{onSet("uriDataExtent")} }
	public var uriString: String?                  { didSet{onSet("uriString")} }

	// data
	public private (set) var dataType = DataType.real32F
	public var data = DataView()
	public var grad = DataView()
	public var history = [DataView]()

	// update weights handler
	public typealias GradientFunction = (DeviceStream) throws -> Void
	public private(set) var updateGradient: GradientFunction!
	public private(set) var updateStream: DeviceStream!
	public func setGradientUpdateFunction(using stream: DeviceStream,
	                                      fn: @escaping GradientFunction) {
		self.updateStream = stream
		updateGradient = fn

		// add to model learned parameter list for training updates
		model.learnedParameters[namePath] = self
	}

	// local
	private var uriDataArray: DataArray?
	private var uriDataShape: Shape?
	private var dataError: DataView!

	//----------------------------------------------------------------------------
	// addAccessors
	public override func addAccessors() {
		super.addAccessors()
		addAccessor(name: "fillMethod",
		            get: { [unowned self] in self.fillMethod },
		            set: { [unowned self] in self.fillMethod = $0 })
		addAccessor(name: "fillIndexStartAt",
		            get: { [unowned self] in self.fillIndexStartAt },
		            set: { [unowned self] in self.fillIndexStartAt = $0 })
		addAccessor(name: "fillMean",
		            get: { [unowned self] in self.fillMean },
		            set: { [unowned self] in self.fillMean = $0 })
		addAccessor(name: "fillRange",
		            get: { [unowned self] in self.fillRange },
		            set: { [unowned self] in self.fillRange = $0 })
		addAccessor(name: "fillStd",
		            get: { [unowned self] in self.fillStd },
		            set: { [unowned self] in self.fillStd = $0 })
		addAccessor(name: "fillValue",
		            get: { [unowned self] in self.fillValue },
		            set: { [unowned self] in self.fillValue = $0 })
		addAccessor(name: "fillVarianceNorm",
		            get: { [unowned self] in self.fillVarianceNorm },
		            set: { [unowned self] in self.fillVarianceNorm = $0 })
		addAccessor(name: "learningRateScale",
		            get: { [unowned self] in self.learningRateScale },
		            set: { [unowned self] in self.learningRateScale = $0 })
		addAccessor(name: "seed",
		            get: { [unowned self] in self.seed },
		            set: { [unowned self] in self.seed = $0 })
		addAccessor(name: "uri",
		            get: { [unowned self] in self.uri },
		            set: { [unowned self] in self.uri = $0 })
		addAccessor(name: "uriDataIsColMajor",
		            get: { [unowned self] in self.uriDataIsColMajor },
		            set: { [unowned self] in self.uriDataIsColMajor = $0 })
		addAccessor(name: "uriDataType",
		            get: { [unowned self] in self.uriDataType },
		            set: { [unowned self] in self.uriDataType = $0 })
		addAccessor(name: "uriDataExtent",
		            get: { [unowned self] in self.uriDataExtent },
		            set: { [unowned self] in self.uriDataExtent = $0 })
		addAccessor(name: "uriString",
		            get: { [unowned self] in self.uriString },
		            set: { [unowned self] in self.uriString = $0 })
	}

	//----------------------------------------------------------------------------
	// copy
	public override func copy(from other: Properties) {
		super.copy(from: other)
		let other = other as! LearnedParameter

		if willLog(level: .diagnostic) {
			diagnostic(
				"\(copyString) \(namePath)(\(data.dataArray.trackingId))" +
				"\(setText(" <--- ", color: .blue))" +
				"\(other.namePath)(\(other.data.dataArray.trackingId)) " +
				"elements: \(other.data.dataArray.elementCount)",
				categories: .dataCopy)
		}

		data = other.data
	}

	//----------------------------------------------------------------------------
	// setup
	// TODO: download uri data
	public override func setup(taskGroup: TaskGroup) throws {
		try super.setup(taskGroup: taskGroup)
		guard fillRange.count == 2 && fillRange[0] <= fillRange[1] else {
			writeLog("fillRange requires lower and upper bounds where lower <= upper")
			throw ModelError.setupFailed
		}
	}

	//----------------------------------------------------------------------------
	// setupData
	//  Load existing weights if specified
	public func setupData(dataType: DataType, using stream: DeviceStream) throws {
		// set type
		self.dataType = dataType

		// the element count will be non zero if this object is a copy
		guard data.elementCount == 0 else { return }

		// convenience string
		if uri == nil, let string = uriString {
			uri = Uri(string: string)
			properties["uri"]!.isGenerated = true
		}

		// load from url if there is one
		guard let url = try uri?.getURL() else { return }
		let array = try DataArray(dataType: uriDataType, contentsOf: url)

		// determine buffer shape
		var shape: Shape
		if let uriDataExtent = uriDataExtent {
			if uriDataExtent.count == 1 {
				shape = Shape(count: array.elementCount)
			} else if uriDataExtent.count == 2 {
				shape = Shape(rows: uriDataExtent[0], cols: uriDataExtent[1],
				              colMajor: uriDataIsColMajor)
			} else {
				shape = Shape(extent: uriDataExtent, colMajor: uriDataIsColMajor)
			}
			uriDataShape = shape

		} else {
			// if no extent, then assume it's a vector
			shape = Shape(count: array.elementCount)
		}

		guard shape.elementCount == array.elementCount else {
			writeLog("Parameter mismatch. uriDataExtent.elementCount = " +
				"\(shape.elementCount), \(uri!.string) elementCount = " +
				"\(array.elementCount)")
			throw ModelError.setupFailed
		}

		// create source view
		let uriData = DataView(shape: shape, dataType: uriDataType, dataArray: array)

		// assign output transforming type and layout if needed
		data = try DataView(from: uriData, asDataType: dataType,
			                  asShape: Shape(extent: shape.extent),
			                  using: stream)
		data.log = currentLog
		data.name = namePath

		if willLog(level: .diagnostic) {
			diagnostic("loaded \(namePath)\(parent!.namespaceName) " +
				"\(shape.extent.description) (\(shape.elementCount)) " +
				"from \(url.absoluteString)", categories: [.setup, .setupForward])
		}
	}

	//----------------------------------------------------------------------------
	// setExtent
	//  Either set or reshape the extent of the weights buffer.
	public func setExtent(_ extent: [Int], using stream: DeviceStream) throws {
		let newShape = Shape(extent: extent)

		if data.shape.elementCount > 0 {
			// you can change shape of existing data but not overall size
			guard data.shape.elementCount == newShape.elementCount else {
				writeLog("The parameter buffer cannot be resized from " +
					"\(data.shape.extent) to \(extent)")
				throw ModelError.setupFailed
			}
			if uriDataShape != nil && uriDataShape!.extent != extent {
				writeLog("The parameter buffer extent \(data.shape.extent) does not " +
					"match expected \(extent)")
				throw ModelError.setupFailed

			} else if data.extent != extent {
				// reinterpret linear weights buffer shape and set Function for logging
				data = DataView(shape: newShape,
				                dataType: data.dataType,
				                dataArray: data.dataArray,
					              name: namePath, log: currentLog)

				if willLog(level: .diagnostic) {
					diagnostic(
						"LearnedParameter(\(trackingId)) reshape " +
							"data.dataArray(\(data.dataArray.trackingId))",
						categories: [.setup, .setupForward])
				}
			}
		} else {
			// create requested storage and set the Function to support logging
			data = DataView(shape: newShape, dataType: dataType,
				              name: namePath, log: currentLog)

			if willLog(level: .diagnostic) {
				diagnostic(
					"LearnedParameter(\(trackingId)) setExtent create " +
						"data.dataArray(\(data.dataArray.trackingId))",
					categories: [.setup, .setupForward])
			}

			// initialize
			switch fillMethod {
				// buffers are zeroed by default
			case .zero: break
			case .constant:
				try stream.fill(data: &data, with: fillValue)

			case .indexed:
				try stream.fillWithIndex(data: &data, startingAt: fillIndexStartAt)

			case .gaussian:
				try stream.fillGaussian(data: &data, mean: fillMean, std: fillStd, seed: seed)

			case .msra:
				try stream.fillMSRA(data: &data,
					varianceNorm: fillVarianceNorm ?? .fanOut, seed: seed)

			case .uniform:
				try stream.fillUniform(data: &data, range: fillRange[0]...fillRange[1], seed: seed)

			case .xavier:
				// TODO: revisit. curand eats a huge amount of gpu memory and seems slower
				// and has poorer results than doing it on the cpu
				try cpuFillXavier(data: &data, varianceNorm: fillVarianceNorm ?? .fanIn, seed: seed)

//				try stream.fillXavier(data: &data,
//					varianceNorm: fillVarianceNorm ?? .fanIn, seed: seed)
			}
		}

		dataError = DataView(count: 1, dataType: data.dataType)
	}

	//----------------------------------------------------------------------------
	// validate_data
	public func validate_data(using stream: DeviceStream) throws {
		try stream.validate(data: data, hasRangeError: &dataError!)
		if try dataError.get() as Bool {
			writeLog("[\(setText("RANGE", color: .red))]" +
				" \(namePath).data is out of range")
			raise(SIGINT)
		}
	}

	// validate_diff
	public func validate_diff(using stream: DeviceStream) throws {
		try stream.validate(data: grad, hasRangeError: &dataError!)
		if try dataError.get() as Bool {
			writeLog("[\(setText("RANGE", color: .red))]" +
				" \(namePath).grad is out of range")
			raise(SIGINT)
		}
	}

} // LearnedParameter
