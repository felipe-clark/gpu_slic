#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "../include/slic.h"



int main(int argc, char** argv)
{
    if(argc !=2)
    {
        printf("Invalid number of parameters (2 expected)\n");
        return -1;
    }

    char* imageName = argv[1];

    cv::Mat image;
    image = cv::imread(imageName, cv::IMREAD_COLOR);

    if(!image.data)
    {
        printf("Could not open image\n");
        return -2;
    }

    cv::Size size(width, height);
    cv::Mat resized;
    cv::resize(image, resized, size);

    if(!resized.isContinuous())
    {
        printf("OpenCV is being difficult. Sorry :,(. Suiciding.\n");
        return -3;
    }

    cv::imwrite("./resized_image.tif", resized);

    cv::Mat lab_image;
    cv::cvtColor(resized, lab_image, cv::COLOR_BGR2Lab);

    printf("Bytes: %lu\n", lab_image.total()*lab_image.channels());

    // Preparations for Kernel invokation
    unsigned char* d_output;
    unsigned char* d_image;
    cv::Mat h_result = lab_image.clone();
    int byte_size = 3 * width * height;
    cudaMalloc(&d_output, byte_size);
    cudaMalloc(&d_image, byte_size);

    cudaMemcpy(d_image, lab_image.data, byte_size, cudaMemcpyHostToDevice);

    spixel_data* h_spixel_data = (spixel_data*)malloc(spixel_size * spixel_size * sizeof(spixel_data));
    initialize_centers(h_spixel_data);

    ownership_data* h_ownership_data = (ownership_data*)malloc(width * height * sizeof(ownership_data));
    initialize_ownership(h_ownership_data);




    // -------------------- The Kernel magic --------------------

    // Configure how to launch the Matrix Add CUDA Kernel
    dim3 threadsPerBlock( 32, 32 ) ;

    // Block split as recommended in the assignment sheet
    int blockPerGridX = (width + threadsPerBlock.x-1)/threadsPerBlock.x;
    int blockPerGridY = (height + threadsPerBlock.y-1)/threadsPerBlock.y;
    int blockPerGridZ = 1;

    dim3 blocksPerGrid( blockPerGridX, blockPerGridY,  blockPerGridZ );

    kernelOverPixels<<<blocksPerGrid, threadsPerBlock>>>(d_image, d_output);

    cudaMemcpy(h_result.data, d_output, byte_size, cudaMemcpyDeviceToHost);
    cudaDeviceReset();

    cv::Mat rgb_result_image;

    test_mark_spixel_centers(h_result.data, h_spixel_data);
    test_block_spixels(h_result.data, h_ownership_data);

    cv::cvtColor(h_result, rgb_result_image, cv::COLOR_Lab2BGR);

    cv::imwrite("./processed_image.jpg", rgb_result_image);

    printf("SUCCESS!\n");

}


void initialize_centers(spixel_data* spx_data)
{
    for (int i = 0; i < spixel_width; i++)
    {
        for(int j = 0; j < spixel_height; j++)
        {
            int index = j * spixel_width + i;
            spx_data[index].l = 127;
            spx_data[index].a = 127;
            spx_data[index].b = 127;
            spx_data[index].x = (2 * spixel_size * i + spixel_size)/2;
            spx_data[index].y = (2 * spixel_size * j + spixel_size)/2;
        }
    }
}

void initialize_ownership(ownership_data* h_ownership_data)
{
    for (int x = 0; x < width; x++)
    {
        for(int y = 0; y < height; y++)
        {
            int own_index = y * width + x;


            int i = x/spixel_size;
            int j = y/spixel_size;

            h_ownership_data[own_index].i = i;
            h_ownership_data[own_index].j = j;
        }
    }
}

void test_mark_spixel_centers(unsigned char* h_image, const spixel_data* spx_data)
{
    for (int i = 0; i < spixel_width; i++)
    {
        for(int j = 0; j < spixel_height; j++)
        {
            int index = j * spixel_width + i;
            int x = spx_data[index].x;
            int y = spx_data[index].y;

            int img_index = 3 * (y * width + x);

            h_image[img_index] = 0;
        }
    }
}

void test_block_spixels(unsigned char* h_image, ownership_data* h_ownership_data)
{
    for (int x = 0; x < width; x++)
    {
        for(int y = 0; y < height; y++)
        {
            int own_index = y * width + x;
            int img_index = 3 * own_index;

            h_image[img_index+1] = h_ownership_data[own_index].i*10;
            h_image[img_index+2] = h_ownership_data[own_index].j*10;
        }
    }
}
