// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/*
Functionality for swapping optimizer tensors to/from (NVMe) storage devices.
*/

#include <condition_variable>
#include <memory>
#include "deepspeed_aio_thread.h"
#include "deepspeed_pin_tensor.h"
#include <torch/torch.h> // Include PyTorch
#include <chrono>
#include <cstring>    // For memset
#include <omp.h>      // OpenMP for parallelism
#include <iostream>
#include <cmath> // For std::isnan
#define DIST_OPT_NUM_OMP_THREADS 16
#define DIST_OPT_NUM_ALIGNMENT 16

static std::unordered_map<std::string, std::vector<size_t>> compressedOffsets;
static std::unordered_map<std::string, size_t> uncompressedSizes;
static std::unordered_map<std::string, size_t> finalFileSizes;

struct deepspeed_aio_handle_t {
    std::unique_ptr<struct aio_context> _aio_ctxt;
    const bool _single_submit;
    const bool _overlap_events;
    const int _num_threads;
    deepspeed_aio_config_t _aio_config;

    std::vector<std::shared_ptr<struct deepspeed_aio_thread_t>> _thread_contexts;
    std::vector<std::thread> _threads;
    int _num_pending_ops;
    std::unique_ptr<struct deepspeed_pin_tensor_t> _pinned_tensor_mgr;
    bool _enable_compression = false;
    char* decomp_temp_buffer = nullptr;
    std::vector<char*> cBuffs;;
    size_t _max_cBuffSize = 0;
    size_t _largest_tensor_bytes = 0;
    
    deepspeed_aio_handle_t(const int block_size,
                           const int queue_depth,
                           const bool single_submit,
                           const bool overlap_events,
                           const int num_threads,
                           const size_t largest_tensor_bytes = 0,
                           std::string lock_name = "",
                           const bool enable_compression = false);

    ~deepspeed_aio_handle_t();

    const int get_block_size() const;
    const int get_queue_depth() const;
    const bool get_single_submit() const;
    const bool get_overlap_events() const;
    const int get_thread_count() const;

    int read(torch::Tensor& buffer, const char* filename, const bool validate);

    int write(const torch::Tensor& buffer, const char* filename, const bool validate);

    int pread(const torch::Tensor& buffer,
              const char* filename,
              const bool validate,
              const bool async,
              int batch_len = -1);

    size_t pwrite(const torch::Tensor& buffer,
               const char* filename,
               const bool validate,
               const bool async,
               int batch_len = -1);

    int sync_pread(torch::Tensor& buffer, const char* filename);

    int sync_pwrite(const torch::Tensor& buffer, const char* filename);

    int async_pread(torch::Tensor& buffer, const char* filename, int batch_len = -1);

    size_t async_pwrite(const torch::Tensor& buffer, const char* filename, int batch_len = -1);

    // TODO: Make API's args to be shape and dtype.
    torch::Tensor new_cpu_locked_tensor(const size_t num_elem, const torch::Tensor& example_tensor);

    bool free_cpu_locked_tensor(torch::Tensor&);

    int wait();

    void _stop_threads();

    void _schedule_aio_work(std::shared_ptr<struct io_op_desc_t> scheduled_op);

    std::shared_ptr<struct io_op_desc_t> _wait_for_aio_work();

    bool _is_valid_parallel_aio_op(const bool read_op, const long long int num_bytes);

    uint8_t* compressTensor(const torch::Tensor& buffer, size_t &compressed_size,  const std::string& filename);
    size_t decompressTensor(torch::Tensor& decompressedBuffer, char* compressedBuffer, const std::string& filename);

    // int get_num_pending() {
    //     return _num_pending_ops;
    // };

    // void print_pending();
};
