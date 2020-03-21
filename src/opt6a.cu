#include <cmath>
#include <cstdio>
#include "../include/slic.h"

__device__ __constant__ float slic_factor;

void initializeSlicFactor()
{
    const float * slic_factor_hp = &slic_factor_h;
    cudaError_t cudaStatus = cudaMemcpyToSymbol(slic_factor, slic_factor_hp, sizeof(float));
}

__global__ void k_cumulativeCountOrig(const pix_data* d_pix_data, const own_data* d_own_data, spx_data* d_spx_data)
{
    //if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0)
    //{
	    //printf("k\n");
    //}

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (y < pix_height && x < pix_width) 
    {
        int pix_index = y * pix_width + x;
        int i = d_own_data[pix_index].i;
        int j = d_own_data[pix_index].j;
        int spx_index = j * spx_width + i;

        atomicAdd(&(d_spx_data[spx_index].l_acc), d_pix_data[pix_index].l);
        atomicAdd(&(d_spx_data[spx_index].a_acc), d_pix_data[pix_index].a);
        atomicAdd(&(d_spx_data[spx_index].b_acc), d_pix_data[pix_index].b);
        atomicAdd(&(d_spx_data[spx_index].num), 1);
    }
}

__global__ void k_cumulativeCountOpt1(const pix_data* d_pix_data, const own_data* d_own_data, spx_data* d_spx_data)
{
    bool debug = false;
    if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0)
    {
	    debug = true;
	    //printf("K\n");
    }

    // If we do 16 instead of 8, only have enough memory for a short, not an int,
    // and 16*32*255 does not fit in a short
    __shared__ int acc[4][3][3][8][32]; //LAB+count, 3x3 neighbors, 8x32 values
    const int arraySize = 4 * 3 * 3;
    const int dimensions = 8 * 32;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = (blockIdx.y * blockDim.y + threadIdx.y) / 2;
    int sx = threadIdx.x;
    int sy = y % 8;
    int cc = threadIdx.y % 2;

    for (int nx=0;nx<3;++nx) for (int ny=0;ny<3;++ny) for(int c=cc; c<4; c+=2) acc[c][ny][nx][sy][sx]=0;

    __syncthreads();

    int i_center = blockIdx.x * blockDim.x / spx_size;
    int j_center = (blockIdx.y * blockDim.y / 2) / spx_size;

    if (cc==0)
    {
        int pix_index = y * pix_width + x;
        int i = d_own_data[pix_index].i;
        int j = d_own_data[pix_index].j;
        int nx = (i<i_center) ? 0 : ((i>i_center) ? 2 : 1);
        int ny = (j<j_center) ? 0 : ((j>j_center) ? 2 : 1);
        acc[0][ny][nx][sy][sx] = d_pix_data[pix_index].l;
        acc[1][ny][nx][sy][sx] = d_pix_data[pix_index].a;
        acc[2][ny][nx][sy][sx] = d_pix_data[pix_index].b;
        acc[3][ny][nx][sy][sx] = 1;
    }
   
    __syncthreads();

    int* accptr = (int*)acc;

    // Collapse over X and Y
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    for (int step=32*8/2; step>0; step /= 2)
    {
	// step = 32 dimensions = 256 arraySize = 36
	int locationIndex = tid % step; // 0..31
	int threadGroup = tid / step; // 0..7

	//int maxThreadGroup = blockDim.x * blockDim.y / step; // 8
	int maxThreadGroup = 512 / step; // 8

	int maxLoopIndex = (arraySize + maxThreadGroup - 1) / maxThreadGroup; // 43/8 = 5

	for (int loopIndex=0; loopIndex<maxLoopIndex; loopIndex++)
        {
	    int innerIndex = loopIndex * maxThreadGroup + threadGroup; //0 8 16 24 32 + (0..7) --> 0..39
	    if (innerIndex >= arraySize) continue; 
            *(accptr + (innerIndex*dimensions + locationIndex)) += 
                *(accptr + (innerIndex*dimensions + locationIndex + step));
        }
	__syncthreads();
    }

    if (tid != 0) return;

    for (int ny=0; ny<3; ny++)
    {
        int j = j_center + ny - 1;
	if (j<0 || j>=spx_height) continue;
        for (int nx=0; nx<3; nx++)
        {
            int i = i_center + nx - 1;
            if (i<0 || i>=spx_width) continue;

            int spx_index = j * spx_width + i;

	    atomicAdd(&(d_spx_data[spx_index].l_acc), (int)acc[0][ny][nx][0][0]);
            atomicAdd(&(d_spx_data[spx_index].a_acc), (int)acc[1][ny][nx][0][0]);
            atomicAdd(&(d_spx_data[spx_index].b_acc), (int)acc[2][ny][nx][0][0]);
            atomicAdd(&(d_spx_data[spx_index].num),   (int)acc[3][ny][nx][0][0]);
        }
    }
}

__global__ void k_averaging(spx_data* d_spx_data)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < spx_width && j < spx_height)
    {
        int spx_index = j * spx_width + i;
        d_spx_data[spx_index].l = d_spx_data[spx_index].l_acc / d_spx_data[spx_index].num;
        d_spx_data[spx_index].a = d_spx_data[spx_index].a_acc / d_spx_data[spx_index].num;
        d_spx_data[spx_index].b = d_spx_data[spx_index].b_acc / d_spx_data[spx_index].num;
    }
}

__global__ void k_ownership(const pix_data* d_pix_data, own_data* d_own_data, const spx_data* d_spx_data)
{
    float min_dist = 10E99;// max_float;
    int min_i = 0;
    int min_j = 0;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (y < pix_height && x < pix_width) 
    {
        int pix_index = y * pix_width + x;
        int i_center = x/spx_size;
        int j_center = y/spx_size;

        int l = d_pix_data[pix_index].l;
        int a = d_pix_data[pix_index].a;
        int b = d_pix_data[pix_index].b;

        for (int i = i_center - window_size; i <= i_center + window_size; i++)
        {
            if (i < 0 || i >= spx_width) continue;

            for(int j = j_center - window_size; j <= j_center + window_size; j++)
            {
                if (j < 0 || j >= spx_height) continue;

                int spx_index = j * spx_width + i;
                int l_dist = l-(int)(d_spx_data[spx_index].l);
                l_dist *= l_dist;
                int a_dist = a-(int)(d_spx_data[spx_index].a);
                a_dist *= a_dist;
                int b_dist = b-(int)(d_spx_data[spx_index].b);
                b_dist *= b_dist;
                int dlab = l_dist + a_dist + b_dist;

                int x_dist = x-(int)d_spx_data[spx_index].x;
                x_dist *= x_dist;
                int y_dist = y-(int)d_spx_data[spx_index].y;
                y_dist *= y_dist;
                int dxy = x_dist + y_dist;

                float D = dlab + slic_factor * dxy;

                if (D < min_dist)
                {
                    min_dist = D;
                    min_i = i;
                    min_j = j;
                }
            }
        }

        d_own_data[pix_index].i = min_i;
        d_own_data[pix_index].j = min_j;

        //d_own_data[pix_index].i = (i_center / 4) * 4;
        //d_own_data[pix_index].j = (j_center / 4) * 4;
    }
}

__global__ void k_reset(spx_data* d_spx_data)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < spx_width && j < spx_height)
    {
        int spx_index = j * spx_width + i;
        d_spx_data[spx_index].l_acc = 0;
        d_spx_data[spx_index].a_acc = 0;
	d_spx_data[spx_index].b_acc = 0;
        d_spx_data[spx_index].num = 0;
    }
}
