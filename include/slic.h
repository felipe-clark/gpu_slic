#ifndef __SLIC__
#define __SLIC__

#include <cmath>
#include <limits>

// Search window of superpixels
// This value must be final 1 due to
// constraints of the implementation
const int window_size = 1;

// Image size, measured in pixels
const int pix_width = 4096;
const int pix_height = 2048;

// Superpixel size, and image size measured in superpixels
const int spx_size = 128; //64 x 64 // Must be divisible by kernel invocation block size
const int spx_width = pix_width/spx_size;
const int spx_height = pix_height/spx_size;

const int slic_m = 10;//10;
const int slic_n = pix_width * pix_height;
const int slic_k = spx_width * spx_height;
const float slic_s = sqrt((float)slic_n/(float)slic_k);
//const float slic_factor_h = (float)slic_m / slic_s;
const float slic_factor_h = 2.5*2.5*slic_m*slic_m/(spx_size*spx_size);
const float max_float_h = std::numeric_limits<float>::max();

// Pixel
// Every pixel of the original image is
// broken down into the three components
// of the LAB color space
struct pix_data
{
    unsigned char l;
    unsigned char a;
    unsigned char b;
};

// Ownership
// Every pixel of the original image is
// "owned" by a cluster centroid
struct own_data
{
    // Coordinates of the centroid in the
    // super pixel matrix
    int i;
    int j;

    public:
    bool isValid() { return i>=0; }
    int getLabel() { return j * spx_width + i; }
    void setLabel(int label)
    {
        j = label / spx_width;
        i = label % spx_width;
    }
    bool operator==(const own_data& other)
    {
        return (this->i == other.i) && (this->j == other.j);
    }
};

// Superpixel
// Every superpixel has a corresponding
// x, y coordinate in the original image
// plane and has averaged LAB components
// taken from all its "owned" pixels.
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
void initializeSlicFactor();
void initialize_own(own_data* h_own_data);
void initialize_n_own(own_data* h_n_own_data);
void initialize_spx(spx_data* h_spx_data);
void enforce_label_connectivity(own_data* o_own_data, const int width,
    const int height, own_data* n_own_data, int n_spx);

// Kernels
#define k_cumulativeCount k_cumulativeCountOpt1
__global__ void k_cumulativeCountOrig(const pix_data* d_pix_data, const own_data* d_own_data, spx_data* d_spx_data);
__global__ void k_cumulativeCountOpt1(const pix_data* d_pix_data, const own_data* d_own_data, spx_data* d_spx_data);
__global__ void k_averaging(spx_data* d_spx_data);
__global__ void k_ownership(const pix_data* d_pix_data, own_data* d_own_data, const spx_data* d_spx_data);
__global__ void k_reset(spx_data* d_spx_data);
// Test Kernels
__global__ void k_test_darkenImage(const pix_data* d_pix_input, pix_data* d_pix_output);

// Colorizers
void color_solid(pix_data* h_pix_data, const own_data* h_own_data, const spx_data* h_spx_data);
void color_borders(pix_data* h_pix_data, const own_data* h_own_data, const spx_data* h_spx_data);
// Test Colorizers
void test_color_own(pix_data* h_pix_data, const own_data* h_own_data, const spx_data* h_spx_data);
void test_color_spx(pix_data* h_pix_data, const own_data* h_own_data, const spx_data* h_spx_data);

#endif
