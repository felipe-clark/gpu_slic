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
} spixel_data;

void initialize_centers(spixel_data* spx_data);
void test_mark_spixel_centers(unsigned char* h_image, const spixel_data* spx_data);

__global__ void kernelOverPixels(unsigned char* d_image, unsigned char* d_output);

#endif