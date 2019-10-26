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
import Foundation

//----------------------------------------------------------------------------
// gemm
//    Row major matrix multiply.
// A(m x k) * B(k x n) -> C(m x n)
public func cpuGemm(alpha: Double, transA: TransposeOp, matrixA: DataView,
                    transB: TransposeOp, matrixB: DataView,
                    beta: Double, matrixC: inout DataView) {
		// make sure we are doing a 2D operation
//		assert(matrixA.rank == 2 || matrixB.rank == 2 || matrixC.rank == 2)
	
//		let m = (transA == .noTranspose) ? matrixA.extent[0] : matrixA.extent[1]
//		let k = (transA == .noTranspose) ? matrixA.extent[1] : matrixA.extent[0]
//		let n = (transB == .noTranspose) ? matrixB.extent[1] : matrixB.extent[0]
//		let rowStrideA = Int32(matrixA.strides[0])
//		let rowStrideB = Int32(matrixB.strides[0])
//		let rowStrideC = Int32(matrixC.strides[0])
//		
}

//------------------------------------------------------------------------------
// cpuAsum
public func cpuAsum(x: DataView, result: inout DataView) throws {
//	assert(result.rank == 1)
	
	if x.shape.isContiguous {
		switch x.dataType {
		case .real16F:
			let x = try x.roReal16F()
			let result = try result.rwReal16F()
			result[0] = x.reduce(Float16(0)) {
				let a = $0
				let b = Float16(abs(Float($1)))
				return a + b
			}
			
		case .real32F:
			let x = try x.roReal32F()
			let result = try result.rwReal32F()
			result[0] = x.reduce(0) { $0 + abs($1) }
			
		case .real64F:
			let x = try x.roReal64F()
			let result = try result.rwReal64F()
			result[0] = x.reduce(0) { $0 + abs($1) }
			
		default: fatalError("not supported")
		}
	} else {
		fatalError("not implemented")
	}
}

//------------------------------------------------------------------------------
// cpuExpand labels
public func cpuExpand(labels: DataView, to expanded: inout DataView) throws {
	assert(labels.dataType == expanded.dataType)
//	assert(labels.rank == 1 && expanded.rank == 2)
	try cpuFill(data: &expanded, with: 0)
	
	// work function
	func setValues<T: AnyNumber>(_ result: inout DataView, _ type: T.Type) throws {
		let pLabels   = try labels.ro(type: T.self)
		let pExpanded = try result.rw(type: T.self)
		for row in 0..<result.extent[0] {
			let labelIndex = row * labels.strides[0]
			let rowIndex   = row * result.strides[0]
			let itemIndex  = pLabels[labelIndex].asInt
			pExpanded[rowIndex + itemIndex] = T(any: 1)
		}
	}

	switch expanded.dataType {
	case .real8U:  try setValues(&expanded, UInt8.self)
	case .real16U: try setValues(&expanded, UInt16.self)
	case .real16I: try setValues(&expanded, Int16.self)
	case .real32I: try setValues(&expanded, Int32.self)
	case .real16F: try setValues(&expanded, Float16.self)
	case .real32F: try setValues(&expanded, Float.self)
	case .real64F: try setValues(&expanded, Double.self)
	}
}

//------------------------------------------------------------------------------
// cpuCompare
//	This is used by the Accuracy Element
//	TODO: write kernels for each compute service
public func cpuCompare(maxIndexOf data: DataView, with labels: DataView,
                       result outData: inout DataView) throws {
	assert(data.dataType == labels.dataType &&
		data.dataType == outData.dataType)
	
	// TODO: others not implemented
//	assert(data.rank == 2 && data.strides[1] == 1 &&
//				 labels.rank == 1 && outData.rank == 1)
	let dataRowStride = data.strides[0]
	
	// work function
	func setValues<T: Comparable & AnyNumber>(
		_ result: inout DataView, _ type: T.Type)	throws
	{
		let pData   = try data.ro(type: T.self)
		let pLabels = try labels.ro(type: T.self)
		let pResult = try result.rw(type: T.self)
		
		for row in 0..<data.extent[0] {
			// find the index of the max value
			let rowOffset = row * dataRowStride
			var maxIndex = 0
			for col in 1..<data.extent[1] {
				if pData[rowOffset + col] > pData[rowOffset + maxIndex] {
					maxIndex = col
				}
			}
			
			// compare with label
			pResult[row] = pLabels[row].asInt == maxIndex ? T(any: 1) : T(any: 0)
		}
	}
	
	switch data.dataType {
	case .real8U:  try setValues(&outData, UInt8.self)
	case .real16U: try setValues(&outData, UInt16.self)
	case .real16I: try setValues(&outData, Int16.self)
	case .real32I: try setValues(&outData, Int32.self)
	case .real16F: try setValues(&outData, Float16.self)
	case .real32F: try setValues(&outData, Float.self)
	case .real64F: try setValues(&outData, Double.self)
	}
}

//------------------------------------------------------------------------------
// axpy
//	y[i] = alpha * x[i] + y[i]
public func cpuAxpy(alpha: Double, x: DataView, y: inout DataView) throws {
//	assert(x.rank == 1 && y.rank == 1 && x.extent[0] == y.extent[0])

	if x.shape.isContiguous {
		switch x.dataType {
		case .real16F:
			let x = try x.roReal16F()
			let y = try y.rwReal16F()
			let alpha = Float(alpha)
			for i in 0..<y.count { y[i] = Float16(alpha * Float(x[i])) + y[i] }
			
		case .real32F:
			let x = try x.roReal32F()
			let y = try y.rwReal32F()
			let alpha = Float(alpha)
			for i in 0..<y.count { y[i] = alpha * x[i] + y[i] }
			
		case .real64F:
			let x = try x.roReal64F()
			let y = try y.rwReal64F()
			for i in 0..<y.count { y[i] = alpha * x[i] + y[i] }
			
		default: fatalError("not supported")
		}
	} else {
		fatalError("not implemented")
	}
}
