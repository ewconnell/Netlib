// swift-tools-version:4.0
import PackageDescription

let package = Package(
	name: "Netlib",
	products: [
		.library(name: "Netlib", targets: ["Netlib"])
	],
	targets: [
		.target(name: "ImageCodecs"),
		.target(name: "Netlib", dependencies: ["ImageCodecs"]),
		.target(name: "diagnosticExample", dependencies: ["Netlib"]),
		.target(name: "trainCodeModel", dependencies: ["Netlib"]),
		.target(name: "trainXmlModel", dependencies: ["Netlib"]),
		.target(name: "vgg16Example", dependencies: ["Netlib"]),
	]
)
