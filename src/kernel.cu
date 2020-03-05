#include "../include/slic.h"

__global__ void k_cumulativeCount(const unsigned char* d_pix_data, const own_data* d_own_data, spx_data* d_spx_data)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (y < pix_height && x < pix_width) 
    {
        int own_index = y * pix_width + x;
        int pix_index = 3 * own_index;
        int i = d_own_data[own_index].i;
        int j = d_own_data[own_index].j;
        int spx_idx = j * spx_width + i;

        atomicAdd(&(d_spx_data[spx_idx].l_acc), d_pix_data[pix_index + 0]);
        atomicAdd(&(d_spx_data[spx_idx].a_acc), d_pix_data[pix_index + 1]);
        atomicAdd(&(d_spx_data[spx_idx].b_acc), d_pix_data[pix_index + 2]);
        atomicAdd(&(d_spx_data[spx_idx].num), 1);
    }
}

__global__ void k_averaging(spx_data* d_spx_data)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < spx_width && j < spx_height)
    {
        int spx_idx = j * spx_width + i;
        d_spx_data[spx_idx].l = d_spx_data[spx_idx].l_acc / d_spx_data[spx_idx].num;
        d_spx_data[spx_idx].a = d_spx_data[spx_idx].a_acc / d_spx_data[spx_idx].num;
        d_spx_data[spx_idx].b = d_spx_data[spx_idx].b_acc / d_spx_data[spx_idx].num;
    }
}