// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/*
Copyright 2020 The Microsoft DeepSpeed Team
Licensed under the MIT license.

Functionality for swapping optimizer tensors to/from (NVMe) storage devices.
*/

#include "deepspeed_py_aio_handle.h"
#include <unistd.h>

using namespace std;

static void _start_aio_thread(std::shared_ptr<struct deepspeed_aio_thread_t> ctxt) { ctxt->run(); }

deepspeed_aio_handle_t::deepspeed_aio_handle_t(const int block_size,
                                               const int queue_depth,
                                               const bool single_submit,
                                               const bool overlap_events,
                                               const int num_threads,
                                               size_t largest_tensor_bytes,
                                               std::string lock_name,
                                               const bool enable_compression)
    : _aio_ctxt(new aio_context(block_size, queue_depth)),
      _single_submit(single_submit),
      _overlap_events(overlap_events),
      _num_threads(num_threads),
      _aio_config(block_size, queue_depth, single_submit, overlap_events, false, lock_name, getpid()),
      _num_pending_ops(0),
      _largest_tensor_bytes(largest_tensor_bytes),
      _enable_compression(enable_compression),
      _pinned_tensor_mgr(new deepspeed_pin_tensor_t())
{
    // my_id = id_gen++;
    // std::cout << ">>>>>>>>> Starting a new deepspeed_aio_handle_t with ID: " << my_id << std::endl;
    for (auto i = 0; i < num_threads; ++i) {
        _thread_contexts.push_back(std::make_shared<deepspeed_aio_thread_t>(i, _aio_config));
    }

    for (auto& ctxt : _thread_contexts) {
        _threads.push_back(std::thread(_start_aio_thread, ctxt));
    }

    if (decomp_temp_buffer == nullptr) {
        decomp_temp_buffer = static_cast<char*>(aligned_alloc(DIST_OPT_NUM_ALIGNMENT, _largest_tensor_bytes));
    }

    _largest_tensor_bytes = ((_largest_tensor_bytes + DIST_OPT_NUM_OMP_THREADS - 1) / DIST_OPT_NUM_OMP_THREADS)* DIST_OPT_NUM_OMP_THREADS;
    size_t chunkSize = _largest_tensor_bytes / DIST_OPT_NUM_OMP_THREADS;
}

deepspeed_aio_handle_t::~deepspeed_aio_handle_t()
{
    _stop_threads();
    for (auto& thr : _threads) { thr.join(); }
    free(decomp_temp_buffer);
}

const int deepspeed_aio_handle_t::get_block_size() const
{
    return _aio_ctxt ? _aio_ctxt->_block_size : -1;
}

const int deepspeed_aio_handle_t::get_queue_depth() const
{
    return _aio_ctxt ? _aio_ctxt->_queue_depth : -1;
}

const bool deepspeed_aio_handle_t::get_single_submit() const { return _single_submit; }

const bool deepspeed_aio_handle_t::get_overlap_events() const { return _overlap_events; }

const int deepspeed_aio_handle_t::get_thread_count() const { return _num_threads; }

int deepspeed_aio_handle_t::read(torch::Tensor& buffer, const char* filename, const bool validate)
{
    const auto start_time = std::chrono::high_resolution_clock::now();

    assert(_aio_ctxt);

    long long num_file_bytes;
    if (-1 == get_file_size(filename, num_file_bytes)) {
        const auto error_code = errno;
        report_file_error(filename, " fstat for read", error_code);
        return -1;
    }
    assert(static_cast<long long int>(buffer.nbytes()) == num_file_bytes);

    const auto fd = open_file(filename, true);
    if (fd == -1) { return -1; }

    auto read_buffer = (char*)buffer.data_ptr();
    std::unique_ptr<io_xfer_ctxt> xfer_ctxt(new io_xfer_ctxt(fd, 0, num_file_bytes, read_buffer));

    if (_aio_config._overlap_events) {
        do_aio_operation_overlap(true, _aio_ctxt, xfer_ctxt, &_aio_config, nullptr);
    } else {
        do_aio_operation_sequential(true, _aio_ctxt, xfer_ctxt, &_aio_config, nullptr);
    }

    close(fd);
    const std::chrono::duration<double> aio_time =
        std::chrono::high_resolution_clock::now() - start_time;

    if (validate) { validate_aio_operation(true, filename, read_buffer, num_file_bytes); }
    const std::chrono::duration<double> fn_time =
        std::chrono::high_resolution_clock::now() - start_time;
    std::cout << "Elapsed time(usec): "
              << "aio = " << aio_time.count() * 1e6 << " call = " << fn_time.count() * 1e6
              << std::endl;
    return 0;
}

int deepspeed_aio_handle_t::write(const torch::Tensor& buffer,
                                  const char* filename,
                                  const bool validate)
{
    assert(_aio_ctxt);

    const auto start_time = std::chrono::high_resolution_clock::now();

    const auto fd = open_file(filename, false);
    if (fd == -1) { return -1; }

    auto write_buffer = (char*)buffer.data_ptr();
    const auto num_write_bytes = static_cast<long long int>(buffer.nbytes());
    std::unique_ptr<io_xfer_ctxt> xfer_ctxt(new io_xfer_ctxt(fd, 0, num_write_bytes, write_buffer));

    if (_aio_config._overlap_events) {
        do_aio_operation_overlap(false, _aio_ctxt, xfer_ctxt, &_aio_config, nullptr);
    } else {
        do_aio_operation_sequential(false, _aio_ctxt, xfer_ctxt, &_aio_config, nullptr);
    }
    const std::chrono::duration<double> aio_time =
        std::chrono::high_resolution_clock::now() - start_time;

    close(fd);

    if (validate) { validate_aio_operation(false, filename, write_buffer, num_write_bytes); }

    const std::chrono::duration<double> fn_time =
        std::chrono::high_resolution_clock::now() - start_time;
    std::cout << "Elapsed time(usec): "
              << "aio = " << aio_time.count() * 1e6 << " call = " << fn_time.count() * 1e6
              << std::endl;
    return 0;
}

void deepspeed_aio_handle_t::_schedule_aio_work(std::shared_ptr<struct io_op_desc_t> scheduled_op)
{
    // int total_pending = 0;
    for (auto& ctxt : _thread_contexts) {
        {
            std::lock_guard<std::mutex> lock(ctxt->_work_sync._mutex);
            ctxt->_work_queue.push(scheduled_op);
        }
        ctxt->_work_sync._cond_var.notify_one();
        // total_pending += ctxt->_work_queue.size();
    }
    _num_pending_ops++;
    // std::cout << "Scheduling AIO work.... " << " read op " << scheduled_op->_read_op << " filename " << scheduled_op->_filename << my_id << std::endl;
    // print_pending();
    // if (_num_pending_ops != total_pending) {
    //     std::cout << "------ Problem here::: ctx " << my_id << " num pending " << _num_pending_ops << " total pending in work_queue " << total_pending << std::endl;
    // }
}

std::shared_ptr<struct io_op_desc_t> deepspeed_aio_handle_t::_wait_for_aio_work()
{
    std::shared_ptr<struct io_op_desc_t> completed_op = nullptr;
    for (auto& ctxt : _thread_contexts) {
        std::unique_lock<std::mutex> lock(ctxt->_complete_sync._mutex);
        ctxt->_complete_sync._cond_var.wait(lock,
                                            [ctxt] { return !ctxt->_complete_queue.empty(); });
        completed_op = ctxt->_complete_queue.front();
        ctxt->_complete_queue.pop();
    }
    return completed_op;
}

void deepspeed_aio_handle_t::_stop_threads()
{
    assert(0 == _num_pending_ops);
    for (auto& ctxt : _thread_contexts) {
        {
            std::lock_guard<std::mutex> lock(ctxt->_work_sync._mutex);
            ctxt->_time_to_exit = true;
        }
        ctxt->_work_sync._cond_var.notify_one();
    }
}

// void deepspeed_aio_handle_t::print_pending() {
//     for (auto& ctxt : _thread_contexts) {
//         if (!ctxt->_work_queue.empty()) {
//             auto x = ctxt->_work_queue.front();
//             std::cout << " Pending in work queue: read_op " << x->_read_op << " filename " << x->_filename << " num ops " << ctxt->_work_queue.size() << " ctx " << my_id << std::endl;
//         } else 
//             std::cout << " Pending in work queue: EMPTY ctx " << my_id << std::endl;
//         if (!ctxt->_complete_queue.empty()) {
//             auto x = ctxt->_complete_queue.front();
//             std::cout << " Pending in complete queue: read_op " << x->_read_op << " filename " << x->_filename << " num ops " << ctxt->_complete_queue.size() << " ctx " << my_id << std::endl;
//         } else 
//             std::cout << " Pending in complete queue: EMPTY ctx " << my_id << std::endl;
//     }
// }

int deepspeed_aio_handle_t::wait()
{
    // std::cout << "In wait(), got num pending ops as " <<_num_pending_ops << " ctx " << my_id << std::endl;
    // print_pending();
    assert(_num_pending_ops > 0);
    auto num_completed_ops = 0;

    while (_num_pending_ops > 0) {
        auto completed_op = _wait_for_aio_work();

        completed_op->fini();

        close(completed_op->_fd);
        bool is_in_map = compressedOffsets.find(completed_op->_filename) != compressedOffsets.end();
        
        if (_enable_compression && completed_op->_read_op && compressedOffsets.count(completed_op->_filename) == 1) {
            size_t final_size = uncompressedSizes[completed_op->_filename];
            size_t num_file_bytes = compressedOffsets[completed_op->_filename][DIST_OPT_NUM_OMP_THREADS];
            assert((completed_op->_num_bytes <= _largest_tensor_bytes));

            
            if (num_file_bytes != 0) {
                // const auto start_time = std::chrono::high_resolution_clock::now();
                #pragma omp parallel num_threads(DIST_OPT_NUM_OMP_THREADS)
                {
                    size_t thread_id = omp_get_thread_num();
                    size_t num_threads = omp_get_num_threads();
                    size_t chunk_size = num_file_bytes / num_threads;
                    size_t start_offset = thread_id * chunk_size;
                    if (thread_id == num_threads - 1) {
                        chunk_size += num_file_bytes % num_threads;
                    }
                    memcpy(static_cast<char*>(decomp_temp_buffer) + start_offset,
                        static_cast<const char*>(completed_op->data_ptr()) + start_offset,
                        chunk_size);
                }
                size_t decompSize = decompressTensor(completed_op->_buffer, decomp_temp_buffer, completed_op->_filename);
                if (decompSize != final_size) {
                    std::cout << "Filename " << completed_op->_filename << " in compressed had " << num_file_bytes << " only got " << decompSize << " should have had " << final_size << std::endl;
                    exit(-1);
                }
                // const auto stop_time = std::chrono::high_resolution_clock::now();
                // const std::chrono::duration<double> decomp_time = stop_time - start_time;
                // std::cout << "\t\t Decomp_time[" << finalFileSizes[completed_op->_filename] << "] = " << decomp_time.count() << std::endl; 
            }
        } 

        if (completed_op->_validate) {
            validate_aio_operation(completed_op->_read_op,
                                   completed_op->_filename.c_str(),
                                   completed_op->data_ptr(),
                                   _num_threads * completed_op->_num_bytes);
        }
        --_num_pending_ops;
        ++num_completed_ops;
    }

    return num_completed_ops;
}

bool deepspeed_aio_handle_t::_is_valid_parallel_aio_op(const bool read_op,
                                                       const long long int num_bytes)
{
    const auto op_string = read_op ? "Read" : "Write";
    if (num_bytes % get_thread_count()) {
        std::cout << "deepspeed_aio failure: parallel " << op_string << " num_bytes = " << num_bytes
                  << " not divisible by thread count = " << get_thread_count() << std::endl;
        return false;
    }

    return true;
}

int deepspeed_aio_handle_t::pread(const torch::Tensor& buffer,
                                  const char* filename,
                                  const bool validate,
                                  const bool async,
                                  int batch_len)
{
    long long num_file_bytes;
    if (-1 == get_file_size(filename, num_file_bytes)) {
        const auto error_code = errno;
        report_file_error(filename, " fstat for read", error_code);
        return -1;
    }

    if (_enable_compression && finalFileSizes.count(filename) == 1 && num_file_bytes < finalFileSizes[filename]) {
        std::cout << "Compressed file " << filename << " to " << finalFileSizes[filename] << " but got " << num_file_bytes << std::endl;
        exit(-1);
        num_file_bytes = finalFileSizes[filename];
    }

    // const auto buffer_bytes = static_cast<long long int>(buffer.nbytes());
    // if (buffer_bytes != num_file_bytes) {
    //     std::cout << filename << ": buffer nbytes != file bytes " << buffer_bytes
    //               << " != " << num_file_bytes << std::endl;
    // }
    // assert(static_cast<long long int>(buffer.nbytes()) == num_file_bytes);

    assert((num_file_bytes % _num_threads) == 0);

    if (!_is_valid_parallel_aio_op(true, num_file_bytes)) { return -1; }

    const auto fd = open_file(filename, true);
    if (fd == -1) { return -1; }

    auto scheduled_op = std::make_shared<io_op_desc_t>(
        true, buffer, fd, filename, (num_file_bytes / _num_threads), validate, batch_len);

    _schedule_aio_work(scheduled_op);

    if (async) { return 0; }

    return wait();
}


uint8_t* deepspeed_aio_handle_t::compressTensor(const torch::Tensor& buffer, size_t& finalCompressedSize, const std::string& filename) {
    // Ensure the input tensor is contiguous for better performance
    if (!buffer.is_contiguous()) {
        std::cerr << "Input tensor must be contiguous." << std::endl;
        return nullptr;
    }

    size_t numElements = buffer.numel() * buffer.element_size();
    uncompressedSizes[filename] = numElements;
    size_t chunkSize = numElements / DIST_OPT_NUM_OMP_THREADS;
    compressedOffsets[filename] = std::vector<size_t>(DIST_OPT_NUM_OMP_THREADS+1, 0);
    uint8_t* finalCompressedBuffer = static_cast<uint8_t*>(buffer.data_ptr());

    // OpenMP parallel region
    #pragma omp parallel num_threads(DIST_OPT_NUM_OMP_THREADS)
    {
        int thread_id = omp_get_thread_num();
        size_t startIdx = thread_id * chunkSize;
        size_t endIdx = (thread_id == DIST_OPT_NUM_OMP_THREADS - 1) ? numElements : startIdx + chunkSize;

        void* cBuff = cBuffs[thread_id];
        // void *cBuff = static_cast<char*>(malloc(cBuffSize));
        if (!cBuff) {
            #pragma omp critical
            {
                std::cerr << "Failed to allocate memory for thread " << thread_id << std::endl;
            }
        }
    }

    // torch::Tensor decomp_buffer_copy = torch::zeros_like(buffer);

    // size_t decomp_size = decompressTensor(decomp_buffer_copy, (char*)finalCompressedBuffer, filename);
    
    // std::cout << "Compressed ." << filename << ". from " << numElements << " to " << compressedOffsets[filename][DIST_OPT_NUM_OMP_THREADS] << " sum og is " <<   init_sum << " decomp to " << decomp_size << " with sum " << torch::sum(decomp_buffer_copy).item<float>() << std::endl;


    return finalCompressedBuffer;
}


size_t deepspeed_aio_handle_t::decompressTensor(torch::Tensor& decompressedBuffer, char* compressedBuffer, const std::string& filename) {
    // Ensure the decompressed buffer is contiguous for better performance
    if (!decompressedBuffer.is_contiguous()) {
        std::cerr << "Decompressed tensor must be contiguous." << std::endl;
        return 0;
    }

    if (compressedOffsets.find(filename) == compressedOffsets.end()) {
        std::cerr << "No compressed offsets found for filename: " << filename << std::endl;
        return 0;
    }

    size_t numElements = decompressedBuffer.numel() * decompressedBuffer.element_size();
    size_t chunkSize = numElements / DIST_OPT_NUM_OMP_THREADS;
    const auto& offsets = compressedOffsets[filename];
    std::atomic<size_t> totalSize = 0;

    // OpenMP parallel region for decompression
    #pragma omp parallel num_threads(DIST_OPT_NUM_OMP_THREADS)
    {
        int thread_id = omp_get_thread_num();
        size_t startIdx = thread_id * chunkSize;
        size_t endIdx = (thread_id == DIST_OPT_NUM_OMP_THREADS - 1) ? numElements : startIdx + chunkSize;

        // Calculate the offset and size of the compressed chunk
        size_t readOffset = offsets[thread_id];
        size_t compressedSize = offsets[thread_id + 1] - readOffset;
        // assert((compressedSize % DIST_OPT_NUM_ALIGNMENT) == 0);
        // if (thread_id == DIST_OPT_NUM_OMP_THREADS -1 )
        // #pragma omp critical
        // {
        // std::cout << " Decompressing File " << filename << " thread  " << thread_id << " by reading from " << readOffset << " to " << offsets[thread_id + 1] << " output start idx is " << startIdx << " output end offset " << endIdx << std::endl;
        // }

        // Allocate a temporary buffer for decompression
        uint8_t* dBuff = static_cast<uint8_t*>(decompressedBuffer.data_ptr()) + startIdx;
        size_t dBuffSize = (endIdx - startIdx);
    }
    
    
    if (totalSize != uncompressedSizes[filename]) {
        std::cout << "For " << filename << " decomp size is " << totalSize << " which should be " << uncompressedSizes[filename] << std::endl;
        std::cout << "The decompressed offsets do not match." << std::endl;
        exit(-1);
    }
    return totalSize;
}

size_t deepspeed_aio_handle_t::pwrite(const torch::Tensor& buffer,
                                   const char* filename,
                                   bool validate,
                                   const bool async,
                                   int batch_len)
{
    auto num_write_bytes = static_cast<long long int>(buffer.nbytes());
    assert((num_write_bytes % _num_threads) == 0);

    if (!_is_valid_parallel_aio_op(false, num_write_bytes)) { return -1; }

    const auto fd = open_file(filename, false);
    if (fd == -1) { return -1; }

    auto scheduled_op = std::make_shared<io_op_desc_t>(false, buffer, fd, filename, (num_write_bytes / _num_threads), validate, batch_len);
    if (_enable_compression) {
        size_t finalCompressedSize = 0;
        // auto start_time = std::chrono::high_resolution_clock::now();
        uint8_t* buffer_ptr = compressTensor(buffer, finalCompressedSize, filename);
        // const auto stop_time = std::chrono::high_resolution_clock::now();
        // const std::chrono::duration<double> comp_time = stop_time - start_time;
        // std::cout << "\t\t Comp_time[" << num_write_bytes << "] = " << comp_time.count() << std::endl; 

        num_write_bytes = ((finalCompressedSize + _num_threads - 1) / _num_threads) * _num_threads;
        size_t block_size = get_block_size();
        num_write_bytes = ((finalCompressedSize + block_size - 1) / block_size) * block_size;
        finalFileSizes[filename] = num_write_bytes;
        // std::cout << filename << " will write a total of " << num_write_bytes << " each writer thread " << (num_write_bytes / _num_threads) << std::endl;
        // validate = true;
        scheduled_op = std::make_shared<io_op_desc_t>(
            false, buffer_ptr, fd, filename, (num_write_bytes / _num_threads), validate, batch_len);
    } 
    _schedule_aio_work(scheduled_op);
    
    if (async) { return num_write_bytes; }

    return wait();
}

int deepspeed_aio_handle_t::sync_pread(torch::Tensor& buffer, const char* filename)
{
    return pread(buffer, filename, false, false);
}

int deepspeed_aio_handle_t::sync_pwrite(const torch::Tensor& buffer, const char* filename)
{
    return pwrite(buffer, filename, false, false);
}

int deepspeed_aio_handle_t::async_pread(torch::Tensor& buffer, const char* filename, int batch_len)
{
    return pread(buffer, filename, false, true, batch_len);
}

size_t deepspeed_aio_handle_t::async_pwrite(const torch::Tensor& buffer, const char* filename, int batch_len)
{
    return pwrite(buffer, filename, false, true, batch_len);
}

at::Tensor deepspeed_aio_handle_t::new_cpu_locked_tensor(const size_t num_elem,
                                                         const torch::Tensor& example_tensor)
{
    return _pinned_tensor_mgr->alloc(num_elem, example_tensor.scalar_type());
}

bool deepspeed_aio_handle_t::free_cpu_locked_tensor(torch::Tensor& locked_tensor)
{
    return _pinned_tensor_mgr->free(locked_tensor);
}
