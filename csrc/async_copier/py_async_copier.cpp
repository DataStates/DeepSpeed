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
        .def("copy", &async_copier_t::copy, py::call_guard<py::gil_scoped_release>())
        .def("is_complete", &async_copier_t::is_complete, py::call_guard<py::gil_scoped_release>())
        .def("wait", &async_copier_t::wait, py::call_guard<py::gil_scoped_release>());
}
