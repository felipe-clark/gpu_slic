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

__global__ void cummulativeCount(unsigned char* d_image, ownership_data* d_ownership_data, spixel_data* d_spixel_data)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (y < height && x < width) 
    {
        int own_index = y * width + x;
        int img_index = 3 * own_index;
        int spix_i = d_ownership_data[own_index].i;
        int spix_j = d_ownership_data[own_index].j;
        int spix_idx = spix_j * spixel_width + spix_i;

        atomicAdd(&(d_spixel_data[spix_idx].l_acc), d_image[img_index + 0]);
        atomicAdd(&(d_spixel_data[spix_idx].a_acc), d_image[img_index + 1]);
        atomicAdd(&(d_spixel_data[spix_idx].b_acc), d_image[img_index + 2]);
        atomicAdd(&(d_spixel_data[spix_idx].n_pix), 1);
    }
}

__global__ void averaging(spixel_data* d_spixel_data)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < spixel_width && j < spixel_height)
    {
        int idx = j * spixel_width + i;
        d_spixel_data[idx].l = d_spixel_data[idx].l_acc / d_spixel_data[idx].n_pix;
        d_spixel_data[idx].a = d_spixel_data[idx].a_acc / d_spixel_data[idx].n_pix;
        d_spixel_data[idx].b = d_spixel_data[idx].b_acc / d_spixel_data[idx].n_pix;
    }
}