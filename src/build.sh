#!/usr/bin/env bash

nvcc slic.cu kernel.cu test.cu -arch=sm_52 -ccbin clang++-3.8 -o slic.exe `pkg-config --libs opencv --cflags` $1 $2 $3 -Xptxas="-v"