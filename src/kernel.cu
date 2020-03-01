#include "../include/slic.h"

__global__ void kernelOverPixels(unsigned char* d_image, unsigned char* d_output)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (y < height && x < width) 
    {
        int index = 3 * (y * width + x);
        d_output[index] = d_image[(index)] >> 1;
        d_output[index+1] = d_image[index+1];
        d_output[index+2] = d_image[index+2];
    }
}

// __global__ void kernelOverSuperPixels(char* d_image, char* d_output, spixel_data* d_spixel_data)
// {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     int j = blockIdx.y * blockDim.y + threadIdx.y;


// }