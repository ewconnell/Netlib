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
