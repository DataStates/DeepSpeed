// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/*
Functionality for swapping optimizer tensors to/from (NVMe) storage devices.
*/
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "async_copier.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    py::class_<async_copier_t>(m, "async_copier")
        .def(py::init<>())
        .def("copy", &async_copier_t::copy)
        .def("wait", &async_copier_t::wait);
}
