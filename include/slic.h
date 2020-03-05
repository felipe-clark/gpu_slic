#ifndef __SLIC__
#define __SLIC__

// Image size, measured in pixels
const int pix_width = 4096;
const int pix_height = 2048;

// Superpixel size, and image size measured in superpixels
const int spx_size = 64; //64 x 64
const int spx_width = pix_width/spx_size;
const int spx_height = pix_height/spx_size;

// Ownership
struct own_data
{
    int i;
    int j;
};

// Superpixel
struct spx_data
{
    int x;
    int y;
    unsigned char l;
    unsigned char a;
    unsigned char b;
    // TODO: Consider padding

    // Accumulators for superpixel averaging
    int l_acc;
    int a_acc;
    int b_acc;
    int num;
};

// Initialization
void initialize_own(own_data* h_own_data);
void initialize_spx(spx_data* h_spx_data);

// Kernels
__global__ void k_cumulativeCount(const unsigned char* d_pix_data, const own_data* d_own_data, spx_data* d_spx_data);
__global__ void k_averaging(spx_data* d_spx_data);
// Test Kernels
__global__ void k_test_darkenImage(const unsigned char* d_pix_input, unsigned char* d_pix_output);

// Colorizers
void color_solid(unsigned char* h_pix_data, const own_data* h_own_data, const spx_data* h_spx_data);
//void color_borders(unsigned char* h_pix_data, const own_data* h_own_data, const spx_data* h_spx_data);
// Test Colorizers
void test_color_own(unsigned char* h_pix_data, const own_data* h_own_data, const spx_data* h_spx_data);
void test_color_spx(unsigned char* h_pix_data, const own_data* h_own_data, const spx_data* h_spx_data);

#endif