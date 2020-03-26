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

        atomicAdd(&(d_spx_data[spx_index].accum[0]), d_pix_data[pix_index].l);
        atomicAdd(&(d_spx_data[spx_index].accum[1]), d_pix_data[pix_index].a);
        atomicAdd(&(d_spx_data[spx_index].accum[2]), d_pix_data[pix_index].b);
        atomicAdd(&(d_spx_data[spx_index].accum[3]), 1);
        atomicAdd(&(d_spx_data[spx_index].accum[4]), x);
        atomicAdd(&(d_spx_data[spx_index].accum[5]), y);
    }
}

__global__ void k_cumulativeCountOpt1(const pix_data* d_pix_data, const own_data* d_own_data, spx_data* d_spx_data)
{
    // If we do 16 instead of 8, only have enough memory for a short, not an int,
    // and 16*32*255 does not fit in a short
    __shared__ short acc[6][3][3][8][32]; //LAB+count, 3x3 neighbors, 8x32 values
    const int arraySize=6*3*3;
    const int dimensions=8*32;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = (blockIdx.y * blockDim.y + threadIdx.y) / OPT6;
    int sx = threadIdx.x;
    int sy = threadIdx.y / OPT6;

    int cc = threadIdx.y % OPT6;
    int ccs = 0; // 0 or cc ?
    int ccstep = 1; // 1 or OPT6 value ?
    if (cc == 0) {
        for (int nx=0;nx<3;++nx) for (int ny=0;ny<3;++ny) for(int c=ccs;c<6;c+=ccstep) acc[c][ny][nx][sy][sx]=0;
    }
    //__syncthreads(); // Sometimes needed for OPT6

    int i_center = blockIdx.x * blockDim.x / spx_size;
    //int j_center = (blockIdx.y * blockDim.y / 4) / spx_size; //OPT6
    int j_center = y / spx_size;

    if (cc==0) { //OPT6
    int pix_index = y * pix_width + x;
    int i = d_own_data[pix_index].i;
    int j = d_own_data[pix_index].j;
    int nx = (i<i_center) ? 0 : ((i>i_center) ? 2 : 1);
    int ny = (j<j_center) ? 0 : ((j>j_center) ? 2 : 1);
    acc[0][ny][nx][sy][sx] = d_pix_data[pix_index].l;
    acc[1][ny][nx][sy][sx] = d_pix_data[pix_index].a;
    acc[2][ny][nx][sy][sx] = d_pix_data[pix_index].b;
    acc[3][ny][nx][sy][sx] = 1;
    acc[4][ny][nx][sy][sx] = x - i_center * spx_size;
    acc[5][ny][nx][sy][sx] = y - j_center * spx_size;
    } //OPT6
   
    __syncthreads();
	
    short* accptr = (short*)acc;

    // Collapse over X and Y
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    for (int step=32*8/2; step>0; step /= 2)
    {
        int locationIndex = tid % step;
        int threadGroup = tid / step;
		
        //int maxThreadGroup = dimensions / step;
        //int maxThreadGroup = blockDim.x * blockDim.y / step;
	int maxThreadGroup = 32 * 8 * OPT6 / step; //OPT6
	
        int maxLoopIndex = (arraySize + maxThreadGroup - 1) / maxThreadGroup;

        // Divide arraySize (3*3*6=54) by max threadGroup + 1 and that's the loop
        // Actual a = loop index * (max threadGroup + 1) + innerIndex
	
        for (int loopIndex=0; loopIndex<maxLoopIndex; loopIndex++)
        {
            int innerIndex = loopIndex * maxThreadGroup + threadGroup;
            if (innerIndex >= arraySize) continue; 

	    //printf("i %d d %d l %d s %d t %d ts %d\n", innerIndex, dimensions, locationIndex, step,
                //innerIndex*dimensions+locationIndex, innerIndex*dimensions+locationIndex+step);
            *(accptr + (innerIndex*dimensions + locationIndex)) += 
                *(accptr + (innerIndex*dimensions + locationIndex + step));
        }
		
        __syncthreads();
    }

    if (tid != 0) return;
    
    // Now, acc[c][ny][nx][0][0] has the values we need
    for (int ny=0; ny<3; ny++)
    {
        int j = j_center + ny - 1;
		if (j<0 || j>=spx_height) continue;
		
        for (int nx=0; nx<3; nx++)
        {
            int i = i_center + nx - 1;
            if (i<0 || i>=spx_width) continue;

            int spx_index = j * spx_width + i;

            atomicAdd(&(d_spx_data[spx_index].accum[0]), (int)acc[0][ny][nx][0][0]);
            atomicAdd(&(d_spx_data[spx_index].accum[1]), (int)acc[1][ny][nx][0][0]);
            atomicAdd(&(d_spx_data[spx_index].accum[2]), (int)acc[2][ny][nx][0][0]);
            atomicAdd(&(d_spx_data[spx_index].accum[3]), (int)acc[3][ny][nx][0][0]);
            atomicAdd(&(d_spx_data[spx_index].accum[4]), (int)acc[4][ny][nx][0][0] + i_center * spx_size);
            atomicAdd(&(d_spx_data[spx_index].accum[5]), (int)acc[5][ny][nx][0][0] + j_center * spx_size);
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
        d_spx_data[spx_index].l = d_spx_data[spx_index].accum[0] / d_spx_data[spx_index].accum[3];
        d_spx_data[spx_index].a = d_spx_data[spx_index].accum[1] / d_spx_data[spx_index].accum[3];
        d_spx_data[spx_index].b = d_spx_data[spx_index].accum[2] / d_spx_data[spx_index].accum[3];
        d_spx_data[spx_index].x = d_spx_data[spx_index].accum[4] / d_spx_data[spx_index].accum[3];
        d_spx_data[spx_index].y = d_spx_data[spx_index].accum[5] / d_spx_data[spx_index].accum[3];
    }
}

__global__ void k_ownershipOpt(const pix_data* d_pix_data, own_data* d_own_data, const spx_data* d_spx_data)
{
    __shared__ spx_data spx[9 * 32];

    float min_dist = 10E99;// max_float;
    int min_i = 0;
    int min_j = 0;

    int i_sign[9] = {-1, -1, -1, 0, 0, 0, 1, 1, 1};
    int j_sign[9] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};
    

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


        if(threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x % 3 == 0)// &&  threadIdx.y == 0)
        {
            int sh_idx = 0;
            for (int i = i_center - window_size; i <= i_center + window_size; i++) // i = i_center - 1, i_center, i_center + 1
            {
                for(int j = j_center - window_size; j <= j_center + window_size; j++) // j = j_center - 1, j_center, j_center + 1
                {
                    if (j < 0 || j >= spx_height || i < 0 || i > spx_width)
                    {
                        sh_idx++;
                        continue;
                    }

                    int spx_index = j * spx_width + i;

                    // if(threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0)
                    //     printf("%i ::::: %i\n", spx_index, sh_idx);

                    
                    spx[sh_idx + 8*blockIdx.x] = d_spx_data[spx_index];
                    
                    if(blockIdx.x > 0 && (sh_idx == 0 || sh_idx == 1 || sh_idx == 2 || sh_idx == 3 || sh_idx == 4 || sh_idx == 5)) //Why blockIdx.x-1 > 0 crashes?
                        spx[sh_idx+3 + 8*(blockIdx.x-1)] = spx[sh_idx + 8*blockIdx.x];

                    if(blockIdx.x > 0 && (sh_idx == 0 || sh_idx == 1 || sh_idx == 2)) //Why blockIdx.x-1 > 0 crashes?
                        spx[sh_idx+6 + 8*(blockIdx.x-2)] = spx[sh_idx + 8*blockIdx.x];

                    if(blockIdx.x < blockDim.x && (sh_idx == 3 || sh_idx == 4 || sh_idx == 5 || sh_idx == 6 || sh_idx == 7 || sh_idx == 8))
                        spx[sh_idx-3 + 8*(blockIdx.x+1)] = spx[sh_idx + 8*blockIdx.x];

                    if(blockIdx.x < blockDim.x && (sh_idx == 6 || sh_idx == 7 || sh_idx == 8))
                        spx[sh_idx-6 + 8*(blockIdx.x+2)] = spx[sh_idx + 8*blockIdx.x];

                    sh_idx++;
                }
            }
        }

        __syncthreads();

        for(int i=0; i<9; i++)
        {
                int l_dist = l-(int)(spx[i + 8*blockIdx.x].l);
                l_dist *= l_dist;
                int a_dist = a-(int)(spx[i + 8*blockIdx.x].a);
                a_dist *= a_dist;
                int b_dist = b-(int)(spx[i + 8*blockIdx.x].b);
                b_dist *= b_dist;
                int dlab = l_dist + a_dist + b_dist;

                int x_dist = x-(int)spx[i + 8*blockIdx.x].x;
                x_dist *= x_dist;
                int y_dist = y-(int)spx[i + 8*blockIdx.x].y;
                y_dist *= y_dist;
                int dxy = x_dist + y_dist;

                float D = dlab + slic_factor * dxy;

            if (D < min_dist)
            {
                min_dist = D;
                min_i = i_center + i_sign[i]*window_size;
                min_j = j_center + j_sign[i]*window_size;
            }
        }

        d_own_data[pix_index].i = min_i;
        d_own_data[pix_index].j = min_j;
    }
}

__global__ void k_ownershipOrig(const pix_data* d_pix_data, own_data* d_own_data, const spx_data* d_spx_data)
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

__global__ void k_ownershipOpt2(const pix_data* d_pix_data, own_data* d_own_data, const spx_data* d_spx_data)
{
    float min_dist = 10E99;// max_float;
    int min_i = 0;
    int min_j = 0;

    __shared__ int spx[3][3][5]; // Y, X, LABXY

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (y < pix_height && x < pix_width) 
    {
        int pix_index = y * pix_width + x;
        int i_center = x/spx_size;
        int j_center = y/spx_size;

	// Initialize SMEM
        int tid = threadIdx.x + blockDim.x * threadIdx.y;
        int nx = tid % 3;
        tid /= 3;
        int ny = tid % 3;
        tid /= 3;
        if (tid < 5)
        {
            int value;
	    int i = i_center + nx - 1;
	    int j = j_center + ny - 1;
	    if (i<0 || i>=spx_width || j<0 || j>=spx_height)
            {
                value = -1;
            }
	    else
            {
	        int spx_index = j * spx_width + i;
	        const spx_data& spix = d_spx_data[spx_index];
	        switch(tid) //TODO:Get rid of it by using better data struct.?
	        {
                    case 0: value=spix.l; break;
		    case 1: value=spix.a; break;
                    case 2: value=spix.b; break;
		    case 3: value=spix.x; break;
		    case 4: value=spix.y; break;
                }
            }
            spx[ny][nx][tid] = value;
        }
	__syncthreads();

        int l = d_pix_data[pix_index].l;
        int a = d_pix_data[pix_index].a;
        int b = d_pix_data[pix_index].b;

        for (int ny=0; ny<3; ++ny) for (int nx=0; nx<3; ++nx)
        {
                int* spix = spx[ny][nx];
		if (spix[0]==-1) continue;

                int l_dist = l-spix[0];
                l_dist *= l_dist;
                int a_dist = a-spix[1];
                a_dist *= a_dist;
                int b_dist = b-spix[2];
                b_dist *= b_dist;
                int dlab = l_dist + a_dist + b_dist;

                int x_dist = x-spix[3];
                x_dist *= x_dist;
                int y_dist = y-spix[4];
                y_dist *= y_dist;
                int dxy = x_dist + y_dist;

                float D = dlab + slic_factor * dxy;

                if (D < min_dist)
                {
                    min_dist = D;
                    min_i = i_center + nx - 1;
                    min_j = j_center + ny - 1;
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
        d_spx_data[spx_index].accum[0] = 0;
        d_spx_data[spx_index].accum[1] = 0;
	d_spx_data[spx_index].accum[2] = 0;
        d_spx_data[spx_index].accum[3] = 0;
	d_spx_data[spx_index].accum[4] = 0;
        d_spx_data[spx_index].accum[5] = 0;
    }
}
