#!/usr/bin/env bash

nvcc slic.cu kernel.cu continuity.cu test.cu -arch=sm_52 -ccbin clang++-3.8 -ptx `pkg-config --libs opencv --cflags` $1 $2 $3 -Xptxas="-v"
