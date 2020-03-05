 #include "../include/slic.h"

// Test-only kernels and functions

// Kernels:

// Darkens the image (reduces brightness by half)
// To use: cv::Mat m_result = m_lab_image.clone(); Pass clone as d_pix_output
__global__ void k_test_darkenImage(const unsigned char* d_pix_input, unsigned char* d_pix_output)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (y < pix_height && x < pix_width) 
    {
        int pix_index = 3 * (y * pix_width + x);
        d_pix_output[pix_index+0] = d_pix_input[pix_index+0] >> 1;
        d_pix_output[pix_index+1] = d_pix_input[pix_index+1];
        d_pix_output[pix_index+2] = d_pix_input[pix_index+2];
    }
}

// Functions:

// Colors the image according to superpixel ownership
// (leaving brightness intact), resulting in funky colors.
// Useful for debugging superpixel ownership.
void test_color_own(unsigned char* h_pix_data, const own_data* h_own_data, const spx_data* h_spx_data)
{
    for (int x = 0; x < pix_width; x++)
    {
        for(int y = 0; y < pix_height; y++)
        {
            int own_index = y * pix_width + x;
            int pix_index = 3 * own_index;

            h_pix_data[pix_index+1] = h_own_data[own_index].i*10;
            h_pix_data[pix_index+2] = h_own_data[own_index].j*10;
        }
    }
}

// Colors superpixel centers in black
void test_color_spx(unsigned char* h_pix_data, const own_data* h_own_data, const spx_data* h_spx_data)
{
    for (int i = 0; i < spx_width; i++)
    {
        for(int j = 0; j < spx_height; j++)
        {
            int spx_index = j * spx_width + i;
            int x = h_spx_data[spx_index].x;
            int y = h_spx_data[spx_index].y;

            int pix_index = 3 * (y * pix_width + x);

            h_pix_data[pix_index] = 0;
        }
    }
}
