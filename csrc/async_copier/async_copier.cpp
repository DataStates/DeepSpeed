#include "async_copier.h"

int async_copier_t::copy(const std::vector<torch::Tensor>& srcs, const std::vector<torch::Tensor>& dests) {
    int n = srcs.size();
    for (int i=0; i<n; i++) {
        void * src_ptr = static_cast<void *>(srcs[i].data_ptr());
        void * dest_ptr = static_cast<void *>(dests[i].data_ptr());
        size_t src_size = srcs[i].numel()*srcs[i].element_size();
        checkCuda(cudaMemcpyAsync(dest_ptr, src_ptr, src_size, cudaMemcpyDefault, trf_stream));
        std::cout << "Enqueuing " << i << " of size " << src_size << std::endl;
    }
    return 0;
}

int async_copier_t::wait() {
    checkCuda(cudaStreamSynchronize(trf_stream));
    return 0;
}