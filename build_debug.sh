#!/bin/bash
mkdir debug && cd debug
cmake -DCMAKE_BUILD_TYPE=Debug -G "CodeBlocks - Unix Makefiles" ..
make trainXmlModel -j12
