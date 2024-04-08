
#include <stdlib.h>
#include <vector>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>


#define checkCuda(ans) { checkCudaFunc((ans), __FILE__, __LINE__); }
inline void checkCudaFunc(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"========= GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

class async_copier_t {
        cudaStream_t trf_stream;
    public:
        async_copier_t() {
            checkCuda(cudaStreamCreateWithFlags(&trf_stream, cudaStreamNonBlocking));
        };
        ~async_copier_t() {
            wait();
        }
        int copy(const std::vector<uintptr_t>& srcs, const std::vector<uintptr_t>& dests, const std::vector<size_t>& sizes);
        int wait();
        int is_complete();
};
