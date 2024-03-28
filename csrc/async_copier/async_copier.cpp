#include "async_copier.h"

int async_copier_t::copy(const std::vector<uintptr_t>& srcs, const std::vector<uintptr_t>& dests, const std::vector<size_t>& sizes) {
    try {
        // wait();
        size_t n = srcs.size();
        for (int i=0; i<n; i++) {
            void * src_ptr =  reinterpret_cast<void*>(static_cast<uintptr_t>(srcs[i]));
            void * dest_ptr = reinterpret_cast<void*>(static_cast<uintptr_t>(dests[i]));
            size_t src_size = sizes[i];
            checkCuda(cudaMemcpyAsync(dest_ptr, src_ptr, src_size, cudaMemcpyDefault, trf_stream));
            // std::cout << "Enqueuing " << i << " of size " << src_size << std::endl;
        }



        // if (n == 1) {
        //     void * src_ptr =  reinterpret_cast<void*>(static_cast<uintptr_t>(srcs[0]));
        //     void * dest_ptr = reinterpret_cast<void*>(static_cast<uintptr_t>(dests[0]));
        //     size_t src_size = sizes[0];
        //     checkCuda(cudaMemcpyAsync(dest_ptr, src_ptr, src_size, cudaMemcpyDefault, trf_stream));
        //     return 0;
        // }

        // wait();
        // if (graph != nullptr) {
        //     checkCuda(cudaGraphDestroy(graph));        
        //     checkCuda(cudaGraphExecDestroy(instance));
        // }
        // checkCuda(cudaStreamBeginCapture(trf_stream, cudaStreamCaptureModeGlobal));
        // for (int i=0; i<n; i++) {
        //     void * src_ptr =  reinterpret_cast<void*>(static_cast<uintptr_t>(srcs[i]));
        //     void * dest_ptr = reinterpret_cast<void*>(static_cast<uintptr_t>(dests[i]));
        //     size_t src_size = sizes[i];
        //     checkCuda(cudaMemcpyAsync(dest_ptr, src_ptr, src_size, cudaMemcpyDefault, trf_stream));
        //     // checkCuda(cudaLaunchHostFunc(trf_stream, printMessage, nullptr));
        //     std::cout << "Enqueuing " << i << " of size " << src_size << std::endl;
        // }
        // checkCuda(cudaStreamEndCapture(trf_stream, &graph));
        // checkCuda(cudaGraphInstantiate(&instance, graph, NULL, NULL, 0));
        // checkCuda(cudaGraphLaunch(instance, trf_stream));



        // size_t numNodes = 0;
        // cudaGraphNode_t* existingNodes = nullptr;
        // if (graph != nullptr)
        //     cudaGraphGetNodes(graph, existingNodes, &numNodes);
        // if (n != numNodes) {
        //     cudaGraphExecDestroy(instance);
        //     cudaGraphDestroy(graph);
        //     std::cout << "Destroying previous graph containing " << numNodes << " elements, new copy needs " << n << " transfers " << std::endl;
        //     numNodes = 0;
        // }

        // if (numNodes == 0) {
        //     checkCuda(cudaStreamBeginCapture(trf_stream, cudaStreamCaptureModeGlobal));
        //     for (int i=0; i<n; i++) {
        //         void * src_ptr =  reinterpret_cast<void*>(static_cast<uintptr_t>(srcs[i]));
        //         void * dest_ptr = reinterpret_cast<void*>(static_cast<uintptr_t>(dests[i]));
        //         size_t src_size = sizes[i];
        //         checkCuda(cudaMemcpyAsync(dest_ptr, src_ptr, src_size, cudaMemcpyDefault, trf_stream));
        //         // checkCuda(cudaLaunchHostFunc(trf_stream, printMessage, nullptr));
        //         std::cout << "Enqueuing " << i << " of size " << src_size << std::endl;
        //     }
        //     checkCuda(cudaStreamEndCapture(trf_stream, &graph));
        //     checkCuda(cudaGraphInstantiate(&instance, graph, NULL, NULL, 0));
        // } else {
        //     for (int i=0; i<n; i++) {
        //         void * src_ptr =  reinterpret_cast<void*>(static_cast<uintptr_t>(srcs[i]));
        //         void * dest_ptr = reinterpret_cast<void*>(static_cast<uintptr_t>(dests[i]));
        //         size_t src_size = sizes[i];
        //         cudaGraphNode_t node = existingNodes[i];
        //         checkCuda(cudaGraphExecMemcpyNodeSetParams1D(instance, node, dest_ptr, src_ptr, src_size, cudaMemcpyDefault));
        //         std::cout << "Updating " << i << " of size " << src_size << std::endl;
        //     }
        // }
        
        // checkCuda(cudaGraphLaunch(instance, trf_stream));
    } catch (const std::exception& e) {
        std::cout << "Exception in copy " << e.what() << std::endl;
    } catch (...) {
        // Catch any other exceptions
        std::cerr << "Caught unknown exception in copy " << std::endl;
    }
    return 0;
}

int async_copier_t::wait() {
    try {
        checkCuda(cudaStreamSynchronize(trf_stream));
        return 0;
    } catch (const std::exception& e) {
        std::cout << "Exception during wait " << e.what() << std::endl;
    } catch (...) {
        // Catch any other exceptions
        std::cerr << "Caught unknown exception in wait " << std::endl;
    }
}

int async_copier_t::is_complete() {
    try {
        cudaError_t err = cudaStreamQuery(trf_stream);
        if (err == cudaSuccess) { // All tasks in the stream are complete
            return 1;
        } else if (err == cudaErrorNotReady) { // Some tasks in the stream are still pending
            return 0;
        } else {
            // Error occurred
            // Handle the error as needed
            return 0;
        }
    } catch (const std::exception& e) {
        std::cout << "Exception in is_completed " << e.what() << std::endl;
    } catch (...) {
        // Catch any other exceptions
        std::cerr << "Caught unknown exception in iscomplete" << std::endl;
    }
}