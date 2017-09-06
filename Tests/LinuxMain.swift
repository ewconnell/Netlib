import XCTest
@testable import NetlibTests

XCTMain([
	testCase(TestDataView.allTests),
	testCase(TestFill.allTests),
	testCase(TestDatabase.allTests),
	testCase(TestGemm.allTests),
	testCase(TestStream.allTests),
	testCase(TestJson.allTests),
	testCase(TestLog.allTests),
	testCase(TestProvider.allTests),
	testCase(TestXml.allTests),
	testCase(TestProperties.allTests),
	testCase(TestSetup.allTests),
])
