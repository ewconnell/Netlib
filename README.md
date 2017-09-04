# Netlib
Netlib is a high performance Swift based machine learning framework that runs on Linux and MacOS. Netlib fully leverages cuDNN and Cuda, but is not architectually dependent on them. Netlib does not require the use of Python and was designed to minimize external dependencies for portability.

Netlib is a research project written entirely in Swift 4.0, with a small amount of C and Cuda code. Netlib intentionally has no Objective-C dependencies.  
Please refer to the Wiki for more detailed information.

# Swiftness
Netlib has been developed while the Swift language and tools have been evolving. Several design decisions have been made along the way that are not ideally Swifty in order to work around compiler crashes, lldb crashes, runtime library crashes, unimplemented Foundation features on Linux, and performance problems. Most problems have been related to generics, protocols, and protocol extensions. In most cases, implementing objects as Swift classes allowed a work around for problems.

