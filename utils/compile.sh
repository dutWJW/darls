#!/bin/bash

cd distributionSample
mkdir build
cd build
cmake ..
make
cd ../..

# Compile cpp subsampling
cd cpp_wrappers/cpp_subsampling
python3 setup.py build_ext --inplace
cd ../..

cd uniformSample
mkdir build
cd build
cmake ..
make
