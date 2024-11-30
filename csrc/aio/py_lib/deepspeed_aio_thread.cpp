// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/*
Functionality for swapping optimizer tensors to/from (NVMe) storage devices.
*/

#include "deepspeed_aio_thread.h"
#include <chrono>
#if defined(__ENABLE_CANN__)
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/UtilForOpAdapter.h"
#endif

using namespace std;

io_op_desc_t::io_op_desc_t(const bool read_op,
                           const torch::Tensor& buffer,
                           const int fd,
                           const char* filename,
                           const long long int num_bytes,
                           const bool validate,
                           int batch_len)
    : _read_op(read_op),
      _buffer(buffer),
      _fd(fd),
      _filename(filename),
      _num_bytes(num_bytes),
      _validate(validate),
      _batch_len(batch_len)
{
    _cpu_buffer = (_buffer.is_cuda() || _buffer.is_xpu()
#if defined(__ENABLE_CANN__)
                   || torch_npu::utils::is_npu(_buffer)
#endif
                       )
                      ? _buffer.to(torch::kCPU).pin_memory()
                      : _buffer;
    _contiguous_buffer = _cpu_buffer.contiguous();
    _buffer_ptr = nullptr;
}


io_op_desc_t::io_op_desc_t(const bool read_op,
                           uint8_t* buffer_ptr,
                           const int fd,
                           const char* filename,
                           const long long int num_bytes,
                           const bool validate,
                           int batch_len)
    : _read_op(read_op),
      _buffer_ptr(buffer_ptr),
      _fd(fd),
      _filename(filename),
      _num_bytes(num_bytes),
      _validate(validate),
      _batch_len(batch_len)
{
    _buffer = torch::empty(0);
    _cpu_buffer = torch::empty(0);
    _contiguous_buffer = torch::empty(0);

}

char* io_op_desc_t::data_ptr() const { 
    if (_buffer_ptr != nullptr)
        return (char*)_buffer_ptr;
    return (char*)_contiguous_buffer.data_ptr(); 
}

void io_op_desc_t::fini()
{
    if (_read_op && _buffer.is_cuda()) { _buffer.copy_(_cpu_buffer.to(torch::kCUDA)); }
    if (_read_op && _buffer.is_xpu()) { _buffer.copy_(_cpu_buffer.to(torch::kXPU)); }
#if defined(__ENABLE_CANN__)
    if (_read_op && torch_npu::utils::is_npu(_buffer)) {
        auto device = at::Device("npu:0");
        _buffer.copy_(_cpu_buffer.to(device));
    }
#endif
}

deepspeed_aio_thread_t::deepspeed_aio_thread_t(const int tid, deepspeed_aio_config_t& aio_config)
    : _tid(tid),
      _aio_config(aio_config),
      _aio_ctxt(new aio_context(aio_config._block_size, aio_config._queue_depth)),
      _time_to_exit(false)
{
}

deepspeed_aio_thread_t::~deepspeed_aio_thread_t() {}

void deepspeed_aio_thread_t::run()
{
    while (true) {
        std::shared_ptr<struct io_op_desc_t> next_io_op = nullptr;

        {
            std::unique_lock<std::mutex> lock(_work_sync._mutex);
            _work_sync._cond_var.wait(lock,
                                      [this] { return (!_work_queue.empty() || _time_to_exit); });
            if (!_work_queue.empty()) {
                next_io_op = _work_queue.front();
                _work_queue.pop();
            }
        }

        if (next_io_op) {
            const auto base_offset = next_io_op->_num_bytes * _tid;

            std::unique_ptr<io_xfer_ctxt> xfer_ctxt(new io_xfer_ctxt(
                next_io_op->_fd, base_offset, next_io_op->_num_bytes, next_io_op->data_ptr()));

            _aio_config.acquire_ipc_lock();
            auto start = std::chrono::high_resolution_clock::now();
            if (_aio_config._overlap_events) {
                do_aio_operation_overlap(
                    next_io_op->_read_op, _aio_ctxt, xfer_ctxt, &_aio_config, nullptr);
            } else {
                do_aio_operation_sequential(
                    next_io_op->_read_op, _aio_ctxt, xfer_ctxt, &_aio_config, nullptr);
            }

            {
                std::lock_guard<std::mutex> lock(_complete_sync._mutex);
                _complete_queue.push(next_io_op);
            }
            _complete_sync._cond_var.notify_one();
            auto stop = std::chrono::high_resolution_clock::now();

            // if (next_io_op->_read_op == false && next_io_op->_buffer_ptr != nullptr) {
            //     free(next_io_op->_buffer_ptr);
            // }
            
            // if (next_io_op->_read_op == false) {
            //     std::cout << "[BYTES = " << next_io_op->_num_bytes << ", OFFSET = " << base_offset << ", TIME = "<< std::chrono::duration_cast<std::chrono::microseconds>(stop-start).count() << ", ADDR = " << (void*)next_io_op->data_ptr() << "]" << std::endl;
            // }
            _aio_config.release_ipc_lock();
        }

        if (_time_to_exit) { break; }
    }
}
