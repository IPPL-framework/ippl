
#include <cuda.h>
#include <cstdlib>

__global__
void kernel(int sz, double* data)
{
    auto _beg = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = _beg; i < sz; ++i)
        data[i] += static_cast<double>(i);
}

int main()
{
    double* data = nullptr;
    int blocks = 64;
    int grids = 64;
    auto ret = cudaMalloc(&data, blocks * grids * sizeof(double));
    if(ret != cudaSuccess)
        return EXIT_FAILURE;
    kernel<<<grids, blocks>>>(blocks * grids, data);
    cudaDeviceSynchronize();
    return EXIT_SUCCESS;
}
