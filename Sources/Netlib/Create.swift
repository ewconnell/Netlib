//******************************************************************************
//  Created by Edward Connell on 5/5/16
//  Copyright © 2016 Connell Research. All rights reserved.
//

extension Properties {
  //------------------------------------------------------------------------------
  // Create
	public func Create(typeName: String) throws -> ModelObject {
		var type = typeName.components(separatedBy: ".")
		
		var modelObject: ModelObject
		if type.count == 1 {
			guard let createFunction = inventory[type[0]] else {
				writeLog("Failed to create unrecognized type \(typeName)")
				throw CreateError.unrecognizedType(typeName)
			}
			modelObject = createFunction()
			modelObject.currentLog = currentLog

		} else {
			// TODO: add external namespace support
			writeLog("Failed to create unrecognized external type \(typeName)")
			throw CreateError.unrecognizedType(typeName)
		}
		return modelObject
	}
}

public enum CreateError : Error {
	case unrecognizedType(String)
}

private let inventory: [String : () -> ModelObject] =	[
	"Accuracy"         : { return Accuracy() },
	"Activation"       : { return Activation() },
	"BatchNormalize"   : { return BatchNormalize() },
	"ComputePlatform"  : { return ComputePlatform() },
	"Connector"        : { return Connector() },
	"Connection"       : { return Connection() },
	"Convolution"      : { return Convolution() },
	"Database"         : { return Database() },
	"DataCodec"        : { return DataCodec() },
	"DataContainer"    : { return DataContainer() },
	"Default"          : { return Default() },
	"Dropout"          : { return Dropout() },
	"FileList"         : { return FileList() },
	"FullyConnected"   : { return FullyConnected() },
	"Function"         : { return Function() },
	"ImageCodec"       : { return ImageCodec() },
	"ImageFormat"      : { return ImageFormat() },
	"ImageNet"         : { return ImageNet() },
	"ImgNetNode"       : { return ImgNetNode() },
	"Label"            : { return Label() },
	"LearnedParameter" : { return LearnedParameter() },
	"Log"              : { return Log() },
	"LrnCrossChannel"  : { return LrnCrossChannel() },
	"Mnist"            : { return Mnist() },
	"Model"            : { return Model() },
	"Uri"              : { return Uri() },
	"Pooling"          : { return Pooling() },
	"ScaleTransform"   : { return ScaleTransform() },
	"Solver"           : { return Solver() },
	"Softmax"          : { return Softmax() },
	"TaskGroup"        : { return TaskGroup() },
	"Test"             : { return Test() },
	"TestDataSource"   : { return TestDataSource() },
	"TinyImageNet"     : { return TinyImageNet() },
	"TransformList"    : { return TransformList() },
]
