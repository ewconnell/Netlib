import PackageDescription

let package = Package(
	name: "Netlib",
	targets: [
		Target(name: "diagnosticExample", dependencies: [.Target(name: "Netlib")]),
		Target(name: "trainCodeModel", dependencies: [.Target(name: "Netlib")]),
		Target(name: "trainXmlModel", dependencies: [.Target(name: "Netlib")]),
		Target(name: "vgg16Example", dependencies: [.Target(name: "Netlib")]),
		Target(name: "Netlib", dependencies: [.Target(name: "ImageCodecs")]),
		Target(name: "ImageCodecs")
	],
	exclude: ["Sources/CudaKernels"]
)
