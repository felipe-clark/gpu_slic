 #include "../include/slic.h"

// Test-only kernels and functions

// Kernels:

// Darkens the image (reduces brightness by half)
// To use: cv::Mat m_result = m_lab_image.clone(); Pass clone as d_pix_output
__global__ void k_test_darkenImage(const pix_data* d_pix_input, pix_data* d_pix_output)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (y < pix_height && x < pix_width) 
    {
        int pix_index = y * pix_width + x;
        d_pix_output[pix_index].l = d_pix_input[pix_index].l >> 1;
        d_pix_output[pix_index].a = d_pix_input[pix_index].a;
        d_pix_output[pix_index].b = d_pix_input[pix_index].b;
    }
}

// Functions:

// Colors the image according to superpixel ownership
// (leaving brightness intact), resulting in funky colors.
// Useful for debugging superpixel ownership.
void test_color_own(pix_data* h_pix_data, const own_data* h_own_data, const spx_data* h_spx_data)
{
    const int color_step = 100;
    for (int x = 0; x < pix_width; x++)
    {
        for(int y = 0; y < pix_height; y++)
        {
            int pix_index = y * pix_width + x;

            h_pix_data[pix_index].a = h_own_data[pix_index].i*color_step;
            h_pix_data[pix_index].b = h_own_data[pix_index].j*color_step;
        }
    }
}

// Colors superpixel centers in black
void test_color_spx(pix_data* h_pix_data, const own_data* h_own_data, const spx_data* h_spx_data)
{
    for (int i = 0; i < spx_width; i++)
    {
        for(int j = 0; j < spx_height; j++)
        {
            int spx_index = j * spx_width + i;
            int x = h_spx_data[spx_index].x;
            int y = h_spx_data[spx_index].y;

            int pix_index = y * pix_width + x;
            h_pix_data[pix_index].l = 0;
        }
    }
}
