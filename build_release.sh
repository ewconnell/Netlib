#!/bin/bash
mkdir -p release && cd release 
cmake -DCMAKE_BUILD_TYPE=Release -G "CodeBlocks - Unix Makefiles" ..
make trainXmlModel

