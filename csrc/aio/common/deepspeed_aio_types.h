// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/*
Functionality for swapping optimizer tensors to/from (NVMe) storage devices.
*/

#include <libaio.h>
#include <stdlib.h>

#include <string>
#include <vector>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/sync/named_mutex.hpp>
#include <boost/interprocess/sync/named_condition.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <atomic>
#include <fstream>
#include <iostream>
#include <thread>
#include <chrono>
#include <mutex>

using namespace std;
using namespace boost::interprocess;

struct deepspeed_aio_latency_t {
    double _min_usec;
    double _max_usec;
    double _avg_usec;

    void dump(const std::string tag);
    void accumulate(const deepspeed_aio_latency_t&);
    void scale(const float value);
};

struct deepspeed_aio_perf_t {
    deepspeed_aio_latency_t _submit;
    deepspeed_aio_latency_t _complete;
    double _e2e_usec;
    double _e2e_rate_GB;
};

struct deepspeed_aio_config_t {
    const int _block_size;
    const int _queue_depth;
    const bool _single_submit;
    const bool _overlap_events;
    const bool _lock_memory;
    std::string _lock_filename;
    const int _my_pid = -1;
    std::atomic<int> _ipc_thread_counter;
    std::atomic<bool> _ipc_has_lock;
    std::unique_ptr<mapped_region> _ipc_region;          // Mapped region
    std::unique_ptr<boost::interprocess::named_mutex> _ipc_mutex;
    std::unique_ptr<boost::interprocess::named_condition> _ipc_cond;

    deepspeed_aio_config_t();
    deepspeed_aio_config_t(const int block_size,
                           const int queue_depth,
                           const bool single_submit,
                           const bool overlap_events,
                           const bool lock_memory,
                           std::string lock_filename = "",
                           const int my_pid = -1);
    void _initialize_ipc_memory();
    void _set_shared_pid(pid_t pid);
    pid_t _get_shared_pid();
    bool acquire_ipc_lock();
    bool release_ipc_lock();
    ~deepspeed_aio_config_t();
};

struct aio_context {
    io_context_t _io_ctxt;
    std::vector<struct io_event> _io_events;
    std::vector<struct iocb*> _iocbs;
    int _block_size;
    int _queue_depth;

    aio_context(const int block_size, const int queue_depth);
    ~aio_context();
};
