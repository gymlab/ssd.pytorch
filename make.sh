#!/usr/bin/env bash
cd ./rfb_tools/

CUDA_PATH=/usr/local/cuda/

python build.py build_ext --inplace

cd ..
