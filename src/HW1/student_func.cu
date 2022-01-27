#include "utils.h"
#include "device_launch_parameters.h"



const size_t blockWidth = 32; //threads per block on one dimension (32*32 total)



__global__
void rgba_to_greyscale(const uchar4* const rgbaImage,
unsigned char* const greyImage,
size_t numRows, size_t numCols)
{
//Fill in the kernel to convert from color to greyscale
//the mapping from components of a uchar4 to RGBA is:
// .x -> R ; .y -> G ; .z -> B ; .w -> A
//
//The output (greyImage) at each pixel should be the result of
//applying the formula: output = .299f * R + .587f * G + .114f * B;
//Note: We will be ignoring the alpha channel for this conversion



//First create a mapping from the 2D block and grid locations
//to an absolute 2D location in the image, then use that to
//calculate a 1D offset
size_t idx_x = threadIdx.x + blockIdx.x*blockDim.x;
size_t idx_y = threadIdx.y + blockIdx.y*blockDim.y;



if (idx_x >= numRows || idx_y >= numCols) return; //it can happen on the "remainder" block

size_t idxvec = idx_x*numCols + idx_y;
uchar4 rgb_value = rgbaImage[idxvec];
greyImage[idxvec] = (unsigned char)(.299f*rgb_value.x + .587f*rgb_value.y + .114f*rgb_value.z);
}



void your_rgba_to_greyscale(const uchar4 * const h_rgbaImage, uchar4 * const d_rgbaImage,
unsigned char* const d_greyImage, size_t numRows, size_t numCols)
{
//You must fill in the correct sizes for the blockSize and gridSize
//currently only one block with one thread is being launched

const dim3 blockSize(blockWidth,blockWidth, 1);
unsigned int numBlocksX = (unsigned int)(numRows / blockWidth + 1);
unsigned int numBlocksY = (unsigned int)(numCols / blockWidth + 1);
const dim3 gridSize(numBlocksX,numBlocksY, 1);
rgba_to_greyscale<<<gridSize, blockSize>>>(d_rgbaImage, d_greyImage, numRows, numCols);

cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());



}
