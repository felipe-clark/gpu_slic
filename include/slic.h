#ifndef __SLIC__
#define __SLIC__

const int width = 4096;
const int height = 2048;

__global__ void testKernel(unsigned char* d_image, unsigned char* d_output);

#endif