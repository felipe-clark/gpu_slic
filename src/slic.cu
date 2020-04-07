#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "../include/slic.h"
#include <sys/time.h>

int main(int argc, char** argv)
{
    if(argc !=2)
    {
        printf("Invalid number of parameters (please provide image filename)\n");
        return -1;
    }

    char* imageName = argv[1];

    cv::Mat m_image;
    m_image = cv::imread(imageName, cv::IMREAD_COLOR);

    if(!m_image.data)
    {
        printf("Could not open image\n");
        return -2;
    }

    cv::Size size(pix_width, pix_height);
    cv::Mat m_resized;
    cv::resize(m_image, m_resized, size);

    if(!m_resized.isContinuous())
    {
        printf("OpenCV is being difficult. Sorry :,(. Suiciding.\n");
        return -3;
    }

    cv::imwrite("./resized_image.tif", m_resized);

    cv::Mat m_lab_image;
    cv::cvtColor(m_resized, m_lab_image, cv::COLOR_BGR2Lab);

    printf("Bytes: %lu\n", m_lab_image.total()*m_lab_image.channels());

    // Preparations for Kernel invocation
    pix_data* d_pix_data;
    own_data* d_own_data;
    spx_data* d_spx_data;

    int pix_byte_size = pix_width * pix_height * sizeof(pix_data);
    int own_byte_size = pix_width * pix_height * sizeof(own_data);
    int spx_byte_size = spx_width * spx_height * sizeof(spx_data);

    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    err = cudaMalloc(&d_pix_data, pix_byte_size);
    reportError(err, __FILE__, __LINE__);
    err = cudaMalloc(&d_own_data, own_byte_size);
    reportError(err, __FILE__, __LINE__);
    err = cudaMalloc(&d_spx_data, spx_byte_size);
    reportError(err, __FILE__, __LINE__);

    pix_data* h_pix_data = (pix_data*)malloc(pix_byte_size);
    for (int x=0; x<pix_width; x++) for (int y=0; y<pix_height; y++)
    {
        int pix_idx = x + pix_width*y;
	h_pix_data[pix_idx].l = ((pix_original_data*)m_lab_image.data)[pix_idx].l;
	h_pix_data[pix_idx].a = ((pix_original_data*)m_lab_image.data)[pix_idx].a;
	h_pix_data[pix_idx].b = ((pix_original_data*)m_lab_image.data)[pix_idx].b;
    }
    err = cudaMemcpy(d_pix_data, h_pix_data, pix_byte_size, cudaMemcpyHostToDevice);
    reportError(err, __FILE__, __LINE__);

    own_data* h_own_data = (own_data*)malloc(own_byte_size);
    initialize_own(h_own_data);
    err = cudaMemcpy(d_own_data, h_own_data, own_byte_size, cudaMemcpyHostToDevice);
    reportError(err, __FILE__, __LINE__);

    own_data* h_n_own_data = (own_data*)malloc(own_byte_size);
    initialize_n_own(h_n_own_data);

    spx_data* h_spx_data = (spx_data*)malloc(spx_byte_size);
    initialize_spx(h_spx_data);
    err = cudaMemcpy(d_spx_data, h_spx_data, spx_byte_size, cudaMemcpyHostToDevice);
    reportError(err, __FILE__, __LINE__);

    initializeSlicFactor();

    // = (float)slic_m / slic_s
// std::numeric_limits<float>::max()

    // -------------------- The Kernel magic --------------------


    //k_measure<<<dim3(10,10), dim3(32,32)>>>(0,1234);

    // Original cumulativeSum kernel
    dim3 pix_threadsPerBlock( 32, 8 ) ;
    int pix_blockPerGridX = (pix_width + pix_threadsPerBlock.x-1)/pix_threadsPerBlock.x;
    int pix_blockPerGridY = (pix_height + pix_threadsPerBlock.y-1)/pix_threadsPerBlock.y;
    dim3 pix_blocksPerGrid(pix_blockPerGridX, pix_blockPerGridY, 1);

    // Ownership kernel - TODO: Optimize
    //dim3 pix_threadsPerBlockOwn( 32, 32 ) ; // Original
    dim3 pix_threadsPerBlockOwn( 128, 8 ) ; // Optimized
    int pix_blockPerGridXOwn = 32;//(pix_width + pix_threadsPerBlockOwn.x-1)/pix_threadsPerBlockOwn.x;   //32
    int pix_blockPerGridYOwn = 16;//(pix_height + pix_threadsPerBlockOwn.y-1)/pix_threadsPerBlockOwn.y;  //16

    printf("%i || %i\n",pix_blockPerGridXOwn, pix_blockPerGridYOwn);
    dim3 pix_blocksPerGridOwn(pix_blockPerGridXOwn, pix_blockPerGridYOwn, 1);

    // Optimized cumulativeSum kernel
    dim3 pix_threadsPerBlockOpt( 128, 1*OPT6 );
    int pix_blockPerGridXOpt = (pix_width + pix_threadsPerBlockOpt.x-1)/pix_threadsPerBlockOpt.x;
    int pix_blockPerGridYOpt = (pix_height + pix_threadsPerBlockOpt.y-1)/pix_threadsPerBlockOpt.y;
    dim3 pix_blocksPerGridOpt(pix_blockPerGridXOpt, OPT6*(pix_blockPerGridYOpt+pix_at_a_time-1)/pix_at_a_time, 1);
    printf("Y:%d\n", pix_blocksPerGridOpt.y);

    //k_ownership<<<pix_blocksPerGridOwn, pix_threadsPerBlockOwn>>>(d_pix_data, d_own_data, d_spx_data);
    
    k_cumulativeCount<<<pix_blocksPerGridOpt, pix_threadsPerBlockOpt>>>(d_pix_data, d_own_data, d_spx_data
    #ifdef BANKDEBUG
    , true
    #endif
    );

    printf("1\n"); cudaDeviceSynchronize(); //TODO

    // Reset and Averaging kernels
    dim3 spx_threadsPerBlock(32, 32);
    int spx_blockPerGridX = (spx_width + spx_threadsPerBlock.x-1)/spx_threadsPerBlock.x;
    int spx_blockPerGridY = (spx_height + spx_threadsPerBlock.y-1)/spx_threadsPerBlock.y;
    dim3 spx_blocksPerGrid(spx_blockPerGridX, spx_blockPerGridY, 1);

    k_averaging<<<spx_blocksPerGrid, spx_threadsPerBlock>>>(d_spx_data);

    const int iterations = 10;
    cudaDeviceSynchronize();
    double ts_start = getTimestamp();
    for (int i = 0 ; i<iterations; i++)
    {
        k_reset<<<spx_blocksPerGrid, spx_threadsPerBlock>>>(d_spx_data);
        err = cudaGetLastError();
        reportError(err, __FILE__, __LINE__);

        k_ownership<<<pix_blocksPerGridOwn, pix_threadsPerBlockOwn>>>(d_pix_data, d_own_data, d_spx_data);
        err = cudaGetLastError();
        reportError(err, __FILE__, __LINE__);

	    k_cumulativeCount<<<pix_blocksPerGridOpt, pix_threadsPerBlockOpt>>>(d_pix_data, d_own_data, d_spx_data
        #ifdef BANKDEBUG
	    , false
        #endif
        );
        err = cudaGetLastError();
        reportError(err, __FILE__, __LINE__);
        
        //printf("REMOVE THIS BEFORE MEASURING\n"); cudaDeviceSynchronize(); //TODO
        k_averaging<<<spx_blocksPerGrid, spx_threadsPerBlock>>>(d_spx_data);
        err = cudaGetLastError();
        reportError(err, __FILE__, __LINE__);
    }

    cudaDeviceSynchronize();
    double ts_end = getTimestamp();
    printf("Average time %0.9f, total %0.9f iters %d\n", (ts_end - ts_start)/iterations, (ts_end - ts_start), iterations);

    err = cudaMemcpy(h_pix_data, d_pix_data, pix_byte_size, cudaMemcpyDeviceToHost);
    reportError(err, __FILE__, __LINE__);
    err = cudaMemcpy(h_own_data, d_own_data, own_byte_size, cudaMemcpyDeviceToHost);
    reportError(err, __FILE__, __LINE__);
    err = cudaMemcpy(h_spx_data, d_spx_data, spx_byte_size, cudaMemcpyDeviceToHost);
    reportError(err, __FILE__, __LINE__);

    const bool doConnectivity = true;
    if (doConnectivity)
    {
        enforce_label_connectivity(h_own_data, pix_width, pix_height, h_n_own_data, spx_width * spx_height);
        err = cudaMemcpy(d_own_data, h_n_own_data, own_byte_size, cudaMemcpyHostToDevice);
        reportError(err, __FILE__, __LINE__);
    }
    else
    {
        // NO-OP
    }

    k_reset<<<spx_blocksPerGrid, spx_threadsPerBlock>>>(d_spx_data);
    err = cudaGetLastError();
    reportError(err, __FILE__, __LINE__);

    // Has to be original cumulativeCount, because we can't assume window size of 1 after conn. enforcement
    k_cumulativeCountOrig<<<pix_blocksPerGrid, pix_threadsPerBlock>>>(d_pix_data, d_own_data, d_spx_data);
    err = cudaGetLastError();
    reportError(err, __FILE__, __LINE__);

    printf("3\n");
    cudaDeviceSynchronize();
    //TODO
    k_averaging<<<spx_blocksPerGrid, spx_threadsPerBlock>>>(d_spx_data);

    err = cudaMemcpy(h_pix_data, d_pix_data, pix_byte_size, cudaMemcpyDeviceToHost);
    reportError(err, __FILE__, __LINE__);
    err = cudaMemcpy(h_own_data, d_own_data, own_byte_size, cudaMemcpyDeviceToHost);
    reportError(err, __FILE__, __LINE__);
    err = cudaMemcpy(h_spx_data, d_spx_data, spx_byte_size, cudaMemcpyDeviceToHost);
    reportError(err, __FILE__, __LINE__);

    color_solid(h_pix_data, h_own_data, h_spx_data);
    //color_borders(h_pix_data, h_own_data, h_spx_data);
    //test_color_own(h_pix_data, h_own_data, h_spx_data);

    for (int x=0; x<pix_width; x++) for (int y=0; y<pix_height; y++)
    {
        int pix_idx = x + pix_width*y;
	((pix_original_data*)m_lab_image.data)[pix_idx].l = h_pix_data[pix_idx].l;
	((pix_original_data*)m_lab_image.data)[pix_idx].a = h_pix_data[pix_idx].a;
	((pix_original_data*)m_lab_image.data)[pix_idx].b = h_pix_data[pix_idx].b;
    }

    cv::Mat m_rgb_result_image;
    cv::cvtColor(m_lab_image, m_rgb_result_image, cv::COLOR_Lab2BGR);
    cv::imwrite("./processed_image.jpg", m_rgb_result_image);

    cudaDeviceSynchronize();
    cudaDeviceReset();
    printf("SUCCESS!\n");
}

// Initializes superpixel centers to be distributed evenly
void initialize_spx(spx_data* h_spx_data)
{
    for (int i = 0; i < spx_width; i++)
    {
        for(int j = 0; j < spx_height; j++)
        {
            int spx_index = j * spx_width + i;
            h_spx_data[spx_index].l = 127;
            h_spx_data[spx_index].a = 127;
            h_spx_data[spx_index].b = 127;
            h_spx_data[spx_index].x = (2 * spx_size * i + spx_size)/2;
            h_spx_data[spx_index].y = (2 * spx_size * j + spx_size)/2;

	    //for (int ny=0; ny<3; ++ny) for (int nx=0; nx<3; ++nx) {
                h_spx_data[spx_index].accum/*[ny][nx]*/[0] = 0;
                h_spx_data[spx_index].accum/*[ny][nx]*/[1] = 0;
                h_spx_data[spx_index].accum/*[ny][nx]*/[2] = 0;
                h_spx_data[spx_index].accum/*[ny][nx]*/[3] = 0;
                h_spx_data[spx_index].accum/*[ny][nx]*/[4] = 0;
                h_spx_data[spx_index].accum/*[ny][nx]*/[5] = 0;
	    //}
        }
    }
}

// Initializes superpixel ownership to regular squares
void initialize_own(own_data* h_own_data)
{
    for (int x = 0; x < pix_width; x++)
    {
        for(int y = 0; y < pix_height; y++)
        {
            int own_index = y * pix_width + x;

            int i = x/spx_size;
            int j = y/spx_size;

            h_own_data[own_index].i = i;
            h_own_data[own_index].j = j;
        }
    }
}

// Initializes superpixel ownership for continuity enforcement
void initialize_n_own(own_data* h_n_own_data)
{
    for (int x = 0; x < pix_width; x++)
    {
        for(int y = 0; y < pix_height; y++)
        {
            int own_index = y * pix_width + x;

            h_n_own_data[own_index].i = -1;
            h_n_own_data[own_index].j = -1;
        }
    }
}

// Solid colorizer: Paints each superpixel with its average color
void color_solid(pix_data* h_pix_data, const own_data* h_own_data, const spx_data* h_spx_data)
{
    for (int x = 0; x < pix_width; x++)
    {
        for(int y = 0; y < pix_height; y++)
        {
            int pix_index = y * pix_width + x;
            int spx_index = h_own_data[pix_index].j * spx_width + h_own_data[pix_index].i;
            h_pix_data[pix_index].l = h_spx_data[spx_index].l;
            h_pix_data[pix_index].a = h_spx_data[spx_index].a;
            h_pix_data[pix_index].b = h_spx_data[spx_index].b;
        }
    }
}

int get_spx_id(const own_data* h_own_data, int x, int y)
{
    int pix_index = y * pix_width + x;
    return h_own_data[pix_index].j * spx_width + h_own_data[pix_index].i;
}

// Colors the border of superpixels to make it easier to visualize them
void color_borders(pix_data* h_pix_data, const own_data* h_own_data, const spx_data* h_spx_data)
{
    for (int x = 0; x < pix_width; x++)
    {
        for(int y = 0; y < pix_height; y++)
        {
            int pix_index = y * pix_width + x;
            int spx_id = get_spx_id(h_own_data, x, y);

            bool border = false;
            border = border || (x == 0) || spx_id != get_spx_id(h_own_data, x-1, y);
            border = border || (x == pix_width-1) || spx_id != get_spx_id(h_own_data, x+1, y);
            border = border || (y == 0) || spx_id != get_spx_id(h_own_data, x, y-1);
            border = border || (y == pix_height-1) || spx_id != get_spx_id(h_own_data, x, y+1);
            border = border || (x == 0) || (y==0) || spx_id != get_spx_id(h_own_data, x-1, y-1);
            border = border || (x == 0) || (y == pix_height-1) || spx_id != get_spx_id(h_own_data, x-1, y+1);
            border = border || (x == pix_width-1) || (y==0) || spx_id != get_spx_id(h_own_data, x+1, y-1);
            border = border || (x == pix_width-1) || (y == pix_height-1) || spx_id != get_spx_id(h_own_data, x+1, y+1);

            if(border)
            {
                h_pix_data[pix_index].l = 0;
                h_pix_data[pix_index].a = 0;
                h_pix_data[pix_index].b = 0;
            }
        }
    }
}

double getTimestamp()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double) tv.tv_usec/1000000.0 + tv.tv_sec;
}

void reportError(cudaError_t err, const char* file, int line)
{
    if (err != cudaSuccess)
        {
            printf("%s failed at %i\n", file, line);
            printf("%s\n", cudaGetErrorString(err));
            exit(-1);
        }
}