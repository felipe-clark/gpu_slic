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

    cv::Mat lab_image;
    cv::cvtColor(image, lab_image, cv::COLOR_BGR2Lab);

    cv::Size size(width, height);
    cv::Mat resized;
    cv::resize(lab_image, resized, size);

    if(!resized.isContinuous())
    {
        printf("OpenCV is being difficult. Sorry :,(. Suiciding.\n");
        return -3;
    }

    printf("Bytes: %lu\n", resized.total()*resized.channels());

    // Preparations for Kernel invokation
    unsigned char* d_output;
    unsigned char* d_image;
    cv::Mat h_result = resized.clone();
    int byte_size = 3 * width * height;
    cudaMalloc(&d_output, byte_size);
    cudaMalloc(&d_image, byte_size);

    cudaMemcpy(d_image, resized.data, byte_size, cudaMemcpyHostToDevice);

    // Configure how to launch the Matrix Add CUDA Kernel
    dim3 threadsPerBlock( 32, 32 ) ;

    // Block split as recommended in the assignment sheet
    int blockPerGridX = (width + threadsPerBlock.x-1)/threadsPerBlock.x;
    int blockPerGridY = (height + threadsPerBlock.y-1)/threadsPerBlock.y;
    int blockPerGridZ = 1;

    dim3 blocksPerGrid( blockPerGridX, blockPerGridY,  blockPerGridZ );

    testKernel<<<blocksPerGrid, threadsPerBlock>>>(d_image, d_output);

    cudaMemcpy(h_result.data, d_output, byte_size, cudaMemcpyDeviceToHost);
    cudaDeviceReset();

    cv::Mat rgb_result_image;
    cv::cvtColor(h_result, rgb_result_image, cv::COLOR_Lab2BGR);

    cv::imwrite("./Darkened_image.jpg", rgb_result_image);

    printf("SUCCESS!\n");

}
