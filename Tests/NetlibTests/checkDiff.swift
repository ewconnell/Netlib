//******************************************************************************
//  Created by Edward Connell on 11/07/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
import XCTest

public func checkDiff(_ a: String, _ b: String) {
	if a == b { return }
	
	// do string grad
	let count = min(a.characters.count, b.characters.count)
	var aIndex = a.startIndex
	var bIndex = b.startIndex
	
	for _ in 0..<count {
		if a.characters[aIndex] != b.characters[bIndex] {
			let aStart = a.index(aIndex, offsetBy: max(-20, a.distance(from: aIndex, to: a.startIndex)))
			let bStart = b.index(bIndex, offsetBy: max(-20, b.distance(from: bIndex, to: b.startIndex)))
			let aEnd = a.index(aIndex, offsetBy: min(100, a.distance(from: aIndex, to: a.endIndex)))
			let bEnd = b.index(bIndex, offsetBy: min(100, b.distance(from: bIndex, to: b.endIndex)))
			XCTAssert(
				a == b, "\nA-----\n\(a[aStart..<aEnd])\n\nB-----\n\(b[bStart..<bEnd])\n-----")
			break
		}
		aIndex = a.index(after: aIndex)
		bIndex = b.index(after: bIndex)
	}
}

