// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/*
Functionality for swapping optimizer tensors to/from (NVMe) storage devices.
*/

#include <cmath>

#include "deepspeed_aio_utils.h"

using namespace std;

const int c_block_size = 128 * 1024;
const int c_io_queue_depth = 8;

deepspeed_aio_config_t::deepspeed_aio_config_t()
    : _block_size(c_block_size),
      _queue_depth(c_io_queue_depth),
      _single_submit(false),
      _overlap_events(false),
      _lock_memory(false),
      _lock_filename(""),
      _my_pid(-1)
{
    _initialize_ipc_memory();
}

deepspeed_aio_config_t::deepspeed_aio_config_t(const int block_size,
                                               const int queue_depth,
                                               const bool single_submit,
                                               const bool overlap_events,
                                               const bool lock_memory,
                                               std::string lock_filename,
                                               const int my_pid
                                               )
    : _block_size(block_size),
      _queue_depth(queue_depth),
      _single_submit(single_submit),
      _overlap_events(overlap_events),
      _lock_memory(lock_memory),
      _lock_filename(lock_filename),
      _my_pid(my_pid)
{
    _initialize_ipc_memory();
}

deepspeed_aio_config_t::~deepspeed_aio_config_t() {
    if (!_lock_filename.empty()) {
        shared_memory_object::remove(_lock_filename.c_str());
        named_mutex::remove((_lock_filename + "_mutex").c_str());
        named_condition::remove((_lock_filename + "_cond").c_str());
    }
}

void deepspeed_aio_config_t::_initialize_ipc_memory() {
    if (_lock_filename.empty())
        return;
    _ipc_has_lock.store(false);
    _ipc_thread_counter.store(0);
    std::unique_ptr<shared_memory_object> _shm_ipc_obj;  // Shared memory object
    _shm_ipc_obj = std::make_unique<shared_memory_object>(open_or_create, _lock_filename.c_str(), read_write);
    _shm_ipc_obj->truncate(sizeof(pid_t));
    _ipc_region = std::make_unique<mapped_region>(*_shm_ipc_obj, read_write);
    std::memset(_ipc_region->get_address(), -1, sizeof(pid_t));  // Initialize to -1

    _ipc_mutex = std::make_unique<boost::interprocess::named_mutex>(
        boost::interprocess::open_or_create, (_lock_filename + "_mutex").c_str());
    _ipc_cond = std::make_unique<boost::interprocess::named_condition>(
        boost::interprocess::open_or_create, (_lock_filename + "_cond").c_str());
}

// Set process ID in shared memory
void deepspeed_aio_config_t::_set_shared_pid(pid_t pid) {
    pid_t* shared_pid = static_cast<pid_t*>(_ipc_region->get_address());
    *shared_pid = pid;
}

// Get process ID from shared memory
pid_t deepspeed_aio_config_t::_get_shared_pid() {
    pid_t* shared_pid = static_cast<pid_t*>(_ipc_region->get_address());
    return *shared_pid;
}


bool deepspeed_aio_config_t::acquire_ipc_lock() {
    if (_lock_filename.empty())
        return true;
    if (!_ipc_has_lock) {  // Only one thread in the process should try to acquire the lock
        while (true) {
            pid_t current_pid = _get_shared_pid();

            if (current_pid == -1) {
                // No process holds the lock, claim it
                _set_shared_pid(_my_pid);
                _ipc_has_lock.store(true);
                break;
            } else if (current_pid == _my_pid) {
                // Current process already holds the lock
                _ipc_has_lock.store(true);
                break;
            } else {
                // boost::interprocess::scoped_lock<boost::interprocess::named_mutex> lock(*_ipc_mutex);
                // _ipc_cond->wait(lock);
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    }
    _ipc_thread_counter.fetch_add(1);
    return true;
}

bool deepspeed_aio_config_t::release_ipc_lock() {
    if (_lock_filename.empty())
        return true;
    _ipc_thread_counter.fetch_add(-1);
    if (_ipc_thread_counter == 0 && _ipc_has_lock) {
        // Last thread in the process, release the lock
        _set_shared_pid(-1);
        _ipc_has_lock.store(false);
        _ipc_cond->notify_all();
    }
    return true;
}

void deepspeed_aio_latency_t::dump(const std::string tag)
{
    std::cout << tag << _min_usec << " " << _max_usec << " " << _avg_usec << " " << std::endl;
}

void deepspeed_aio_latency_t::accumulate(const struct deepspeed_aio_latency_t& other)
{
    _min_usec += other._min_usec;
    _max_usec += other._max_usec;
    _avg_usec += other._avg_usec;
}

void deepspeed_aio_latency_t::scale(const float scaler)
{
    _min_usec *= scaler;
    _max_usec *= scaler;
    _avg_usec *= scaler;
}

aio_context::aio_context(const int block_size, const int queue_depth)
{
    _block_size = block_size;
    _queue_depth = queue_depth;
    for (auto i = 0; i < queue_depth; ++i) {
        _iocbs.push_back((struct iocb*)calloc(1, sizeof(struct iocb)));
    }
    _io_events.resize(queue_depth);
    io_queue_init(queue_depth, &_io_ctxt);
}

aio_context::~aio_context()
{
    for (auto& iocb : _iocbs) { free(iocb); }
    _io_events.resize(0);
    io_queue_release(_io_ctxt);
}
