#include <cmath>
#include <cstdio>
#include "../include/slic.h"

__device__ __constant__ float slic_factor;

void initializeSlicFactor()
{
    const float * slic_factor_hp = &slic_factor_h;
    cudaError_t cudaStatus = cudaMemcpyToSymbol(slic_factor, slic_factor_hp, sizeof(float));
}

__global__ void k_measure(int* d_device_location, int target)
{
    int accum = threadIdx.x;
    for (int i=1; i<100; i++) for (int j=1; j<1000; j++)
    {
        accum *= j;
	accum = accum ^ (threadIdx.y << j / 100);
	accum += target;
    }
    if (accum == target) *d_device_location = 0;
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

        atomicAdd(&(d_spx_data[spx_index].accum/*[0][0]*/[0]), d_pix_data[pix_index].l);
        atomicAdd(&(d_spx_data[spx_index].accum/*[0][0]*/[1]), d_pix_data[pix_index].a);
        atomicAdd(&(d_spx_data[spx_index].accum/*[0][0]*/[2]), d_pix_data[pix_index].b);
        atomicAdd(&(d_spx_data[spx_index].accum/*[0][0]*/[3]), 1);
        atomicAdd(&(d_spx_data[spx_index].accum/*[0][0]*/[4]), x);
        atomicAdd(&(d_spx_data[spx_index].accum/*[0][0]*/[5]), y);
    }
}

#define dimensions_x 128
#define dimensions_y 1
#define dimensions (dimensions_x * dimensions_y)
#define log2_dimensions_x 7
#define log2_dimensions_y 0
#define log2_dimensions (log2_dimensions_x + log2_dimensions_y)
#define log2_pix_at_a_time 7

#define sums 54
#define log2_pix_width 12
#define const_pix_width 4096
#define log2_spx_size 7
#define log2_spx_width 5
__global__ void k_cumulativeCountOpt1(const pix_data* d_pix_data, const own_data* d_own_data, spx_data* d_spx_data)
{
    //bool debug = (blockIdx.x == 20 && blockIdx.y == 30 && threadIdx.x == 5);
    //if (debug) printf("D\n");

    typedef int itemsToSum[dimensions];
    __shared__ itemsToSum acc[6][3][3]; //LAB+count, 3x3 neighbors, 128 values

    int x = (blockIdx.x << log2_dimensions_x) + threadIdx.x;
    int y = ((blockIdx.y << log2_dimensions_y) + threadIdx.y) << log2_pix_at_a_time;
    int sx = (threadIdx.y << log2_dimensions_x) + threadIdx.x; //thread id

    // Initialize SMEM to 0
    int* accptr = (int*)acc;
    itemsToSum* sumptr = (itemsToSum*)acc;

    #pragma unroll
    for (int i=0; i<sums; ++i) sumptr[i][sx] = 0;

    accptr = (int*)acc;
    
    int i_center = blockIdx.x; // OPT14:  * blockDim.x / spx_size;
    //int j_center = blockIdx.y; // OPT14: y / spx_size;
    //int j_center = y >> log2_spx_size;
    int j_center = y / spx_size;

    int pix_index = (y << log2_pix_width) + x;
    for (int yidx=0; yidx<pix_at_a_time; ++yidx) {
	
        int odata = *((int*)(d_own_data + pix_index));
	own_data od = *((own_data*)(&odata));    
	int i = od.i;
        int j = od.j;
        
	int nx = (i<i_center) ? 0 : ((i>i_center) ? 2 : 1);
        int ny = (j<j_center) ? 0 : ((j>j_center) ? 2 : 1);

        int pdata = *((int*)(d_pix_data + pix_index));
	pix_data pd = *((pix_data*)(&pdata));

	int ayidx=1;
        acc[0][ny][nx][sx] = (int)pd.l
            + (ayidx?(acc[0][ny][nx][sx]):0);
        acc[1][ny][nx][sx] = (int)pd.a
            + (ayidx?(acc[1][ny][nx][sx]):0);
        acc[2][ny][nx][sx] = (int)pd.b
            + (ayidx?(acc[2][ny][nx][sx]):0);
        acc[3][ny][nx][sx] = (int)1
            + (ayidx?(acc[3][ny][nx][sx]):0);
        acc[4][ny][nx][sx] = (int)x
            + (ayidx?(acc[4][ny][nx][sx]):0);
        acc[5][ny][nx][sx] = (int)(y+yidx)
            + (ayidx?(acc[5][ny][nx][sx]):0);
        //if (debug) 
		//printf("yidx:%d ny:%d nx:%d accX:%d, accY:%d\n", yidx, ny, nx, acc[4][ny][nx][sx], acc[5][ny][nx][sx]);
	pix_index += const_pix_width;
    }
   
    __syncthreads();

    // Collapse over X and Y
    for (int log2_step=log2_dimensions-1; log2_step>=0; --log2_step)
    {
	int step = 1 << log2_step;
        int locationIndex = sx % step;
        int threadGroup = sx >> log2_step;
		
	int maxThreadGroup = 1 << (log2_dimensions - log2_step);	
        int maxLoopIndex = (sums + maxThreadGroup - 1) / maxThreadGroup;

        // Divide arraySize (3*3*6=54) by max threadGroup + 1 and that's the loop
        // Actual a = loop index * (max threadGroup + 1) + innerIndex

        // It looks like a lot of unnecessary math (multiplications, etc) is going
        // on below, but all attempts to optimize this lead to slowdowns. Looks like the
        // compiler is doing something smart here.	
        for (int loopIndex=0; loopIndex<maxLoopIndex; loopIndex++)
        {
            int innerIndex = loopIndex * maxThreadGroup + threadGroup;
            if (innerIndex >= sums) continue; 

            *(accptr + ((innerIndex<<log2_dimensions) + locationIndex)) += 
                *(accptr + ((innerIndex<<log2_dimensions) + locationIndex + step));
        }
		
        __syncthreads();
    }

    if (sx >= sums) return;
    int c = sx % 6;
    sx /= 6;
    int nx = sx % 3;
    int ny = sx / 3;

    int j = j_center + ny - 1;
    if (j<0 || j>=spx_height) return;
		
    int i = i_center + nx - 1;
    if (i<0 || i>=spx_width) return;

    int spx_index = (j << log2_spx_width) + i;

    int* accum = (int*)(d_spx_data[spx_index].accum);
    //accum[sx*6 + c] = (int)acc[c][ny][nx][0];
    atomicAdd(accum+c,(int)acc[c][ny][nx][0]);
}


__global__ void k_averaging(spx_data* d_spx_data)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    //bool debug = (i==20 && (j==15 || j==16));
    //bool debug = true;

    if (i < spx_width && j < spx_height)
    {
        int spx_index = j * spx_width + i;
	int num = 0, l = 0, a = 0, b = 0, x = 0, y = 0;
	//for (int ny=0; ny<3; ++ny) for (int nx=0; nx<3; ++nx) 
	//{
            l   += d_spx_data[spx_index].accum/*[ny][nx]*/[0];
            a   += d_spx_data[spx_index].accum/*[ny][nx]*/[1];
            b   += d_spx_data[spx_index].accum/*[ny][nx]*/[2];
            num += d_spx_data[spx_index].accum/*[ny][nx]*/[3];
            x   += d_spx_data[spx_index].accum/*[ny][nx]*/[4];
            y   += d_spx_data[spx_index].accum/*[ny][nx]*/[5];
	//}
	//if (debug) printf("i:%d j:%d l:%d a:%d b:%d num:%d x:%d y:%d\n",
	    //i,j,l/num,a/num,b/num,num,x/num,y/num);
        d_spx_data[spx_index].l = l / num;
        d_spx_data[spx_index].a = a / num;
        d_spx_data[spx_index].b = b / num;
        d_spx_data[spx_index].x = x / num;
        d_spx_data[spx_index].y = y / num;
    }
}

__global__ void k_ownershipOpt(const pix_data* d_pix_data, own_data* d_own_data, const spx_data* d_spx_data)
{
    return; // Does not work after opt14, used to be 9*32
    __shared__ spx_data spx[1 * 1];

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

    __shared__ int spx[4][3][5]; // Y, X, LABXY - [4] in first dimension to minimize shared memory bank conflicts

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (y < pix_height && x < pix_width) 
    {
        int pix_index = y * pix_width + x;
        int i_center = x/spx_size;
        int j_center = y/spx_size;

        // Reading as a single blob
        int lab_data = *((int*)(d_pix_data + pix_index));
        pix_data px_data = *((pix_data*)(&lab_data));   

        unsigned char l = px_data.l;
        unsigned char a = px_data.a;
        unsigned char b = px_data.b;

	    // Initialize SMEM
        int tid = threadIdx.x + blockDim.x * threadIdx.y;
        int nx = tid % 3;
        tid /= 3;
        int ny = tid % 3;
        tid /= 3;
        
        if (tid == 0)
        {
            int vl=-1;
            int va=-1;
            int vb=-1;
            int vx=-1;
            int vy=-1;
	        int i = i_center + nx - 1;
	        int j = j_center + ny - 1;
            
            if (i>=0 && i<spx_width && j>=0 && j<spx_height)
            {
	            int spx_index = j * spx_width + i;
                const spx_data& spix = d_spx_data[spx_index]; //TODO: This is compromising efficiency by 25%! But still the best result

                vl=spix.l;
                va=spix.a;
                vb=spix.b;
                vx=spix.x;
                vy=spix.y;
            }

            spx[ny][nx][0] = vl;
            spx[ny][nx][1] = va;
            spx[ny][nx][2] = vb;
            spx[ny][nx][3] = vx;
            spx[ny][nx][4] = vy;
        }
        
        __syncthreads();

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

        // Writing as a blob
        // This reaches 100% write efficiency.
        int mins = min_i << 0 | min_j <<  8;
        *(int*)(d_own_data  + pix_index) = mins;
    }
}

__global__ void k_ownershipOpt3(const pix_data* d_pix_data, own_data* d_own_data, const spx_data* d_spx_data)
{
    __shared__ int spx[3][3][5]; // Y, X, LABXY

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int i_center = x/spx_size;
    int j_center = y/spx_size;

    // Copy super-pixels  to SMEM
    int tid = threadIdx.x + blockDim.x * threadIdx.y;
    int nx = tid % 3;
    tid /= 3;
    int ny = tid % 3;
    tid /= 3;
    
    if (tid == 0)
    {
        int vl=-1;
        int va=-1;
        int vb=-1;
        int vx=-1;
        int vy=-1;
        int i = i_center + nx - 1;
        int j = j_center + ny - 1;
        
        if (i>=0 && i<spx_width && j>=0 && j<spx_height)
        {
            int spx_index = j * spx_width + i;
            const spx_data& spix = d_spx_data[spx_index];

            vl=spix.l;
            va=spix.a;
            vb=spix.b;
            vx=spix.x;
            vy=spix.y;
        }

        spx[ny][nx][0] = vl;
        spx[ny][nx][1] = va;
        spx[ny][nx][2] = vb;
        spx[ny][nx][3] = vx;
        spx[ny][nx][4] = vy;
    }
    
    __syncthreads();


    #define pix_per_thread 16
    pix_data px[pix_per_thread];
    for (int i=0; i<pix_per_thread; i++)
    {
        int pix_index = ((y*pix_per_thread+i) * pix_width) + x;
        int lab_data = *((int*)(d_pix_data + pix_index));
        pix_data px_data = *((pix_data*)(&lab_data));
        px[i] = px_data;

        // Compute ownership
    
        float min_dist = 10E99;
        int min_i = 0;
        int min_j = 0;
    
        for (int ny=0; ny<3; ++ny)
        { 
            for (int nx=0; nx<3; ++nx)
            {
                int* spix = spx[ny][nx];
                if (spix[0]==-1) continue;

                int l_dist = px[i].l-spix[0];
                l_dist *= l_dist;
                int a_dist = px[i].a-spix[1];
                a_dist *= a_dist;
                int b_dist = px[i].b-spix[2];
                b_dist *= b_dist;
                int dlab = l_dist + a_dist + b_dist;

                int x_dist = x-spix[3];
                x_dist *= x_dist;
                int y_dist = y*pix_per_thread+i-spix[4];
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
        }
        
        // Writing as a blob
        // This reaches 100% write efficiency.
        int mins = min_i << 0 | min_j <<  8;
        *(int*)(d_own_data + pix_index) = mins;         
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
	//for (int ny=0; ny<3; ++ny) for (int nx=0; nx<3; ++nx) {
            d_spx_data[spx_index].accum/*[ny][nx]*/[0] = 0;
            d_spx_data[spx_index].accum/*[ny][nx]*/[1] = 0;
	        d_spx_data[spx_index].accum/*[ny][nx]*/[2] = 0;
            d_spx_data[spx_index].accum/*[ny][nx]*/[3] = 0;
    	    d_spx_data[spx_index].accum/*[ny][nx]*/[4] = 0;
            d_spx_data[spx_index].accum/*[ny][nx]*/[5] = 0;
	//}
    }
}
