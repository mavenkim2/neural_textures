#include <cuda.h>
#include <mma.h>

namespace
{

struct Matrix
{
};

struct Layer
{
    void Backward()
    {
        // nvcuda::wmma help me;
        // matrix mult
        // y = xw + b
        // dy/dx = w
        // dl/dx = dl/dy * dy/dx
        // dl/dx = dl/dy * w
    }
};

} // namespace
