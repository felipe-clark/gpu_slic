#!/usr/bin/env bash

nvcc slic.cu -arch=sm_52 -ccbin clang++-3.8 -o slic `pkg-config --libs opencv --cflags` $1 $2 $3 -Xptxas="-v"