cmake_minimum_required(VERSION 3.5)
project(Netlib)

#####################
# output directories
string(TOLOWER "${CMAKE_BUILD_TYPE}" SWIFT_BUILD_TYPE)
set(TARGET_DIR ${CMAKE_SOURCE_DIR}/.build/${SWIFT_BUILD_TYPE})
set(PLUGINS_DIR ${TARGET_DIR}/PlugIns)
find_package(CUDA QUIET REQUIRED)

#####################
# C modules
set(MODULES_DIR ${PROJECT_SOURCE_DIR}/modules)

file(GLOB_RECURSE MODULES_HEADERS
        ${MODULES_DIR}/*.h
        ${MODULES_DIR}/*.modulemap)

#####################
# CppUtils static library
# swift doesn't know how to build cpp files, so we build separately and link in
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
#
#set(CPP_UTILS_NAME CppUtils)
#
#file(GLOB_RECURSE CPP_UTILS_SOURCES
#        Sources/CppUtils/*.cpp
#        Sources/CppUtils/*.h)
#
#add_library(${CPP_UTILS_NAME} ${CPP_UTILS_SOURCES})
#
#set_target_properties(${CPP_UTILS_NAME} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${TARGET_DIR})
#add_custom_command(TARGET ${CPP_UTILS_NAME} PRE_BUILD
#        COMMAND ${CMAKE_COMMAND} -E make_directory ${TARGET_DIR})

#####################
# swift
#find_program(SWIFT swift)
set(SWIFT_HOME /home/ed/swift/usr)
set(SWIFT ${SWIFT_HOME}/bin/swift)
set(SWIFTC ${SWIFT_HOME}/bin/swiftc)
set(SWIFT_CLANG_INCLUDE ${SWIFT_HOME}/lib/swift/clang/include)

# configuration specific swift flags
set(SWIFT_CONFIG_FLAGS
#        -Xswiftc -warnings-as-errors
        -Xswiftc -j12
        -Xswiftc -swift-version -Xswiftc 4
        -Xswiftc -I/usr/local/include
        -Xswiftc -L/usr/local/lib
        -Xswiftc -I${SWIFT_CLANG_INCLUDE}
        -Xswiftc -I${PROJECT_SOURCE_DIR}/Sources/CudaKernels
        -Xswiftc -I${PROJECT_SOURCE_DIR}/Sources/ImageCodecs
        -Xswiftc -I${MODULES_DIR}/Cuda
        -Xswiftc -I${MODULES_DIR}/Jpeg
        -Xswiftc -I${MODULES_DIR}/Lmdb
        -Xswiftc -I${MODULES_DIR}/Png
        -Xswiftc -I${MODULES_DIR}/Zlib
        -Xswiftc -I${TARGET_DIR}
        -Xswiftc -L${TARGET_DIR}
        -Xlinker -lCudaKernels
        -Xswiftc -I${CUDA_INCLUDE_DIRS}
        -Xlinker -L${CUDA_TOOLKIT_ROOT_DIR}/lib64
        )

if (${TARGET_DIR} MATCHES "debug")
    set(SWIFT_CONFIG_FLAGS ${SWIFT_CONFIG_FLAGS}
            -Xswiftc -DDEBUG -Xcc -DDEBUG
            -Xswiftc -DENABLE_TRACKING
            -Xswiftc -g -Xcc -g -Xcc -O0)
else()
    set(SWIFT_CONFIG_FLAGS ${SWIFT_CONFIG_FLAGS} -Xcc -O3)
#    set(SWIFT_CONFIG_FLAGS ${SWIFT_CONFIG_FLAGS} -Xcc -O3 -Xswiftc -g -Xcc -g)
#    set(SWIFT_CONFIG_FLAGS ${SWIFT_CONFIG_FLAGS} -Xswiftc -DENABLE_TRACKING)
endif ()

# thread sanitizer
#set(SWIFT_CONFIG_FLAGS ${SWIFT_CONFIG_FLAGS} -Xcc -fsanitize=thread -Xcc -g -Xcc -fno-omit-frame-pointer -Xlinker -ltsan)

# address sanitizer
#set(SWIFT_CONFIG_FLAGS ${SWIFT_CONFIG_FLAGS} -Xcc -fsanitize=address -Xcc -fno-omit-frame-pointer -Xcc -g  -Xcc -O1 -Xlinker -lasan)

# memory sanitizer
#set(SWIFT_CONFIG_FLAGS ${SWIFT_CONFIG_FLAGS} -Xcc -fsanitize=memory -Xcc -fsanitize-memory-track-origins -Xcc -fno-omit-frame-pointer -Xcc -g  -Xcc -O1 -Xlinker -lmsan)

#####################
# CudaKernels lib
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
        -Wno-deprecated-gpu-targets
#        -gencode arch=compute_52,code=sm_52
        -gencode arch=compute_61,code=sm_61
        --compiler-options -fPIC)

# for emulator debug -G -g -deviceemu
# for cuda-memcheck  -Xcompiler -rdynamic -lineinfo -G
if (${TARGET_DIR} MATCHES "debug")
    # adding the debug flags seems to make the swift debugger crazy
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -DDEBUG)
endif()

# CudaKernels static library
# swift doesn't know how to build cuda files, so we build separately and link in
set(CUDA_KERNELS_NAME CudaKernels)

file(GLOB_RECURSE CUDA_KERNELS_SOURCES
        Sources/CudaKernels/*.cu
        Sources/CudaKernels/*.h)

cuda_add_library(${CUDA_KERNELS_NAME} ${CUDA_KERNELS_SOURCES})

set_target_properties(${CUDA_KERNELS_NAME} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${TARGET_DIR})
add_custom_command(TARGET ${CUDA_KERNELS_NAME} PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${TARGET_DIR})

#####################
# Netlib modules
set(NETLIB_NAME Netlib)
file(GLOB_RECURSE NETLIB_SOURCES
        ${PROJECT_SOURCE_DIR}/Sources/*.swift
        ${PROJECT_SOURCE_DIR}/Sources/*.c
        ${PROJECT_SOURCE_DIR}/Sources/*.h)

set(NETLIB_FLAGS ${SWIFT_CONFIG_FLAGS})

add_custom_target(${NETLIB_NAME}
        COMMAND ${SWIFT} build -c ${SWIFT_BUILD_TYPE} ${NETLIB_FLAGS}
        SOURCES ${NETLIB_SOURCES} ${MODULES_HEADERS})

add_dependencies(${NETLIB_NAME} ${CUDA_KERNELS_NAME} ${CPP_UTILS_NAME})

#####################
# diagnosticExample app
set(DIAGNOSTIC_EXAMPLE_NAME diagnosticExample)
file(GLOB_RECURSE DIAGNOSTIC_EXAMPLE_SOURCES
        ${PROJECT_SOURCE_DIR}/Sources/diagnosticExample/*.swift)

add_custom_target(${DIAGNOSTIC_EXAMPLE_NAME}
        COMMAND ${SWIFT} build -c ${SWIFT_BUILD_TYPE} ${NETLIB_FLAGS}
        SOURCES ${DIAGNOSTIC_EXAMPLE_SOURCES} ${MODULES_HEADERS})

add_dependencies(${DIAGNOSTIC_EXAMPLE_NAME} ${NETLIB_NAME})


#####################
# trainXmlModel app
set(TRAINXMLMODEL_NAME trainXmlModel)
file(GLOB_RECURSE TRAINXMLMODEL_SOURCES
        ${PROJECT_SOURCE_DIR}/Sources/trainXmlModel/*.swift)

add_custom_target(${TRAINXMLMODEL_NAME}
        COMMAND ${SWIFT} build -c ${SWIFT_BUILD_TYPE} ${NETLIB_FLAGS}
        SOURCES ${TRAINXMLMODEL_SOURCES} ${MODULES_HEADERS})

add_dependencies(${TRAINXMLMODEL_NAME} ${NETLIB_NAME})


#####################
# trainCodeModel app
set(TRAINCODEMODEL_NAME trainCodeModel)
file(GLOB_RECURSE TRAINCODEMODEL_SOURCES
        ${PROJECT_SOURCE_DIR}/Sources/trainCodeModel/*.swift)

add_custom_target(${TRAINCODEMODEL_NAME}
        COMMAND ${SWIFT} build -c ${SWIFT_BUILD_TYPE} ${NETLIB_FLAGS}
        SOURCES ${TRAINCODEMODEL_SOURCES} ${MODULES_HEADERS})

add_dependencies(${TRAINCODEMODEL_NAME} ${NETLIB_NAME})


#####################
# NetlibTests app
set(NETLIB_TESTS_NAME NetlibTests)
file(GLOB_RECURSE NETLIB_TEST_SOURCES
        ${PROJECT_SOURCE_DIR}/Tests/*.swift)

#set(TEST_CASE -s NetlibTests.TestSetup/test_mnistForward)

add_custom_target(${NETLIB_TESTS_NAME}
        COMMAND ${SWIFT} test ${TEST_CASE} ${NETLIB_FLAGS}
        SOURCES ${NETLIB_TEST_SOURCES})

add_dependencies(${NETLIB_TESTS_NAME} ${NETLIB_NAME} ${CUDA_KERNELS_NAME})

#####################
# samples data
add_custom_command(TARGET ${NETLIB_NAME} PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${PROJECT_SOURCE_DIR}/samples ${TARGET_DIR}/samples)

######################
## CudaServicePlugIn module
#set(CUDA_SERVICE_PLUGIN_NAME CudaServicePlugIn)
#file(GLOB_RECURSE CUDA_SERVICE_PLUGIN_SOURCES
#        ${PROJECT_SOURCE_DIR}/Sources/CudaService/*.swift)
#
#set(CUDA_SERVICE_PLUGIN_FLAGS
#        -j12
#        -I/usr/local/include
#        -L/usr/local/lib
#        -I${PROJECT_SOURCE_DIR}/Sources/CudaKernels
#        -I${MODULES_DIR}/Cuda
#        -I${MODULES_DIR}/Jpeg
#        -I${MODULES_DIR}/Lmdb
#        -I${MODULES_DIR}/Png
#        -I${MODULES_DIR}/Zlib
#        -I${PROJECT_SOURCE_DIR}/Sources/ImageCodecs
#        -I${TARGET_DIR}
#        -L${TARGET_DIR}
#        -lCudaKernels
#        -I${CUDA_INCLUDE_DIRS}
#        -L${CUDA_TOOLKIT_ROOT_DIR}/lib64
#        -emit-library
#        -emit-module -module-name ${CUDA_SERVICE_PLUGIN_NAME}
#        -emit-module-path ${TARGET_DIR}/${CUDA_SERVICE_PLUGIN_NAME}.swiftmodule
#        -o ${TARGET_DIR}/lib${CUDA_SERVICE_PLUGIN_NAME}.so)
#
#add_custom_target(${CUDA_SERVICE_PLUGIN_NAME}
#        COMMAND ${SWIFTC} ${CUDA_SERVICE_PLUGIN_FLAGS} ${CUDA_SERVICE_PLUGIN_SOURCES}
#        SOURCES ${CUDA_SERVICE_PLUGIN_SOURCES} ${MODULES_HEADERS})
#
#add_dependencies(${CUDA_SERVICE_PLUGIN_NAME} ${CUDA_KERNELS_NAME} ${NETLIB_NAME})

#add_custom_command(TARGET ${NETLIB_NAME} PRE_BUILD
#        COMMAND ${CMAKE_COMMAND} -E make_directory ${TARGET_DIR}/PlugIns
#        COMMAND ${CMAKE_COMMAND} -E copy ${TARGET_DIR}/CudaService.swiftmodule ${PLUGINS_DIR}/.
#        COMMAND ${CMAKE_COMMAND} -E copy ${TARGET_DIR}/CudaService.swiftdoc ${PLUGINS_DIR}/.)

