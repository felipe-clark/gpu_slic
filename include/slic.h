#ifndef __SLIC__
#define __SLIC__

const int width = 4096;
const int height = 2048;

const int spixel_size = 64; //64 x 64
const int spixel_width = width/spixel_size;
const int spixel_height = height/spixel_size;

typedef struct spixel_data 
{
    int x;
    int y;
    unsigned char l;
    unsigned char a;
    unsigned char b;
    unsigned char unused; // padding;

    // Accumulators for superpixel averaging
    int l_acc;
    int a_acc;
    int b_acc;
    int n_pix;
} spixel_data;

typedef struct ownership_data
{
    int i;
    int j;
} ownership_data;

void initialize_centers(spixel_data* spx_data);
void initialize_ownership(ownership_data* h_ownership_data);

__global__ void kernelOverPixels(unsigned char* d_image, unsigned char* d_output);
__global__ void cummulativeCount(unsigned char* d_image, ownership_data* d_ownership_data, spixel_data* d_spixel_data);
__global__ void averaging(spixel_data* d_spixel_data);

void test_mark_spixel_centers(unsigned char* h_image, const spixel_data* spx_data);
void test_block_spixels(unsigned char* h_image, ownership_data* h_ownership_data, spixel_data* h_spixel_data);

#endif