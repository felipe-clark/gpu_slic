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

        atomicAdd(&(d_spx_data[spx_index].accum[0][0][0]), d_pix_data[pix_index].l);
        atomicAdd(&(d_spx_data[spx_index].accum[0][0][1]), d_pix_data[pix_index].a);
        atomicAdd(&(d_spx_data[spx_index].accum[0][0][2]), d_pix_data[pix_index].b);
        atomicAdd(&(d_spx_data[spx_index].accum[0][0][3]), 1);
        atomicAdd(&(d_spx_data[spx_index].accum[0][0][4]), x);
        atomicAdd(&(d_spx_data[spx_index].accum[0][0][5]), y);
    }
}

__global__ void k_cumulativeCountOpt1(const pix_data* d_pix_data, const own_data* d_own_data, spx_data* d_spx_data)
{
    //bool debug = (blockIdx.x == 20 && blockIdx.y == 30 && threadIdx.x == 5);

    __shared__ int acc[6][3][3][128]; //LAB+count, 3x3 neighbors, 8x32 values
    const int memX = 0; // Extra added to X over 32 to avoid memory bank conflicts
    const int arraySize=6*3*3;
    const int dimensions=128 + memX;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * pix_at_a_time;
    int sx = threadIdx.x;

    for (int nx=0;nx<3;++nx) for (int ny=0;ny<3;++ny) for(int c=0;c<6;++c) acc[c][ny][nx][sx]=0;

    int i_center = blockIdx.x * blockDim.x / spx_size;
    int j_center = y / spx_size;
    //if (debug) printf("x: %d y: %d ic:%d jc: %d\n", x, y, i_center, j_center);

    for (int yidx=0; yidx<pix_at_a_time; ++yidx) {
	if ((y+yidx)>=pix_height) break;
        int pix_index = (y + yidx) * pix_width + x;
        int i = d_own_data[pix_index].i;
        int j = d_own_data[pix_index].j;
        int nx = (i<i_center) ? 0 : ((i>i_center) ? 2 : 1);
        int ny = (j<j_center) ? 0 : ((j>j_center) ? 2 : 1);
	//if (debug && yidx==62) printf("A i: %d j:%d nx: %d ny: %d pi: %d a0: %d\n",i,j,nx,ny,pix_index,
            //acc[0][ny][nx][sx]);
        acc[0][ny][nx][sx] = (int)d_pix_data[pix_index].l
            + (yidx?(acc[0][ny][nx][sx]):0);
	//if (debug && yidx==62) printf("B i: %d j:%d nx: %d ny: %d pi: %d a0: %d\n",i,j,nx,ny,pix_index,
            //acc[0][ny][nx][sx]);
        acc[1][ny][nx][sx] = (int)d_pix_data[pix_index].a
            + (yidx?(acc[1][ny][nx][sx]):0);
        acc[2][ny][nx][sx] = (int)d_pix_data[pix_index].b
            + (yidx?(acc[2][ny][nx][sx]):0);
        acc[3][ny][nx][sx] = (int)1
            + (yidx?(acc[3][ny][nx][sx]):0);
        acc[4][ny][nx][sx] = ((int)x - (i_center * spx_size))
            + (yidx?(acc[4][ny][nx][sx]):0);
        acc[5][ny][nx][sx] = ((int)(y+yidx) - (j_center * spx_size))
            + (yidx?(acc[5][ny][nx][sx]):0);
    }
   
    __syncthreads();
	
    int* accptr = (int*)acc;

    // Collapse over X and Y
    for (int step=128/2; step>0; step /= 2)
    {
        int locationIndex = sx % step;
        int threadGroup = sx / step;
		
	int maxThreadGroup = 128/step;
	
        int maxLoopIndex = (arraySize + maxThreadGroup - 1) / maxThreadGroup;

	//if (debug) printf("S:%d LOC:%d TG:%d/MTG:%d MLI:%d\n",
            //step, locationIndex, threadGroup, maxThreadGroup, maxLoopIndex);

        // Divide arraySize (3*3*6=54) by max threadGroup + 1 and that's the loop
        // Actual a = loop index * (max threadGroup + 1) + innerIndex
	
        for (int loopIndex=0; loopIndex<maxLoopIndex; loopIndex++)
        {
            int innerIndex = loopIndex * maxThreadGroup + threadGroup;
            if (innerIndex >= arraySize) continue; 

	    //printf("i %d d %d l %d s %d t %d ts %d\n", innerIndex, dimensions, locationIndex, step,
                //innerIndex*dimensions+locationIndex, innerIndex*dimensions+locationIndex+step);
	    
	    int loc2 = locationIndex + step;
	    int loc = locationIndex + memX*(locationIndex/128);
	    loc2 += memX*(loc2/128);
            *(accptr + (innerIndex*dimensions + loc)) += 
                *(accptr + (innerIndex*dimensions + loc2));
        }
		
        __syncthreads();
    }

    
    // Now, acc[c][ny][nx][0] has the values we need

    /*
    for (int nx=0;nx<3;nx++) for (int ny=0;ny<3;ny++) for(int c=0;c<6;c++){
    int j = j_center + ny - 1;
    if (j<0 || j>=spx_height) return;
		
    int i = i_center + nx - 1;
    if (i<0 || i>=spx_width) return;

    int spx_index = j * spx_width + i;
	    atomicAdd(&(d_spx_data[spx_index].accum[c]), (int)acc[c][ny][nx][0]);
    }
    return;
    */

    if (sx >= arraySize) return;
    int c = sx % 6;
    sx /= 6;
    int nx = sx % 3;
    int ny = sx / 3;

    int j = j_center + ny - 1;
    if (j<0 || j>=spx_height) return;
		
    int i = i_center + nx - 1;
    if (i<0 || i>=spx_width) return;

    int spx_index = j * spx_width + i;

    //Opt128
    //#pragma unroll
    //for (int adj=0;adj<2;adj++)
    //{
        //atomicAdd(&(d_spx_data[spx_index].accum[c]), (int)acc[c][ny][nx][0]);
        int* accum = (int*)(d_spx_data[spx_index].accum);
	accum[sx*6 + c] = (int)acc[c][ny][nx][0] +
            (c>3 ? (((c==4)?i_center:j_center)*spx_size*acc[3][ny][nx][0]) : 0);
        //atomicAdd(&(d_spx_data[spx_index].accum[c]), (int)acc[c][ny][nx][0] +
            //(c>3 ? (((c==4)?i_center:j_center)*spx_size*acc[3][ny][nx][0]) : 0));
    //}
    
    //if (i_center==30 && j_center==15 && d_spx_data[spx_index].accum[3]>0) printf("ic:%d jc:%d x:%d y:%d, qty:%d\n",i_center,j_center,d_spx_data[spx_index].accum[4],d_spx_data[spx_index].accum[5], d_spx_data[spx_index].accum[3]);
}


__global__ void k_averaging(spx_data* d_spx_data)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < spx_width && j < spx_height)
    {
        int spx_index = j * spx_width + i;
	int num = 0, l = 0, a = 0, b = 0, x = 0, y = 0;
	for (int ny=0; ny<3; ++ny) for (int nx=0; nx<3; ++nx) 
	{
            l   += d_spx_data[spx_index].accum[ny][nx][0];
            a   += d_spx_data[spx_index].accum[ny][nx][1];
            b   += d_spx_data[spx_index].accum[ny][nx][2];
            num += d_spx_data[spx_index].accum[ny][nx][3];
            x   += d_spx_data[spx_index].accum[ny][nx][4];
            y   += d_spx_data[spx_index].accum[ny][nx][5];
	}
        d_spx_data[spx_index].l = l / num;
        d_spx_data[spx_index].a = a / num;
        d_spx_data[spx_index].b = b / num;
        d_spx_data[spx_index].x = x / num;
        d_spx_data[spx_index].y = y / num;
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
    // Shared memory conflict test
    // Removing the "*64" below results in no bank conflicts, so adjacent threads
    // reading adjacent shorts do not cause conflicts.
    //__shared__ unsigned short arr[32 * 2 * 100];
    //int a=arr[threadIdx.x * 64];
    //d_spx_data[0].accum[0]=a;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < spx_width && j < spx_height)
    {
        int spx_index = j * spx_width + i;
	for (int ny=0; ny<3; ++ny) for (int nx=0; nx<3; ++nx) {
            d_spx_data[spx_index].accum[ny][nx][0] = 0;
            d_spx_data[spx_index].accum[ny][nx][1] = 0;
	    d_spx_data[spx_index].accum[ny][nx][2] = 0;
            d_spx_data[spx_index].accum[ny][nx][3] = 0;
    	    d_spx_data[spx_index].accum[ny][nx][4] = 0;
            d_spx_data[spx_index].accum[ny][nx][5] = 0;
	}
    }
}
