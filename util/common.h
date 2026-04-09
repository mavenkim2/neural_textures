#if defined(__CUDACC__)
#define NT_HOST_DEVICE __host__ __device__
#define NT_DEVICE __device__
#else
#define NT_HOST_DEVICE
#define NT_DEVICE
#endif
