
#include "chemistry.h"
#include "memory.h"
#include "raytracing.cuh"

#include <Python.h>
#include <numpy/arrayobject.h>

#include <iostream>

/* @file python_module.cu
 * @brief ASORA Python C-extension module
 *
 * This file contains the wrappers for python to access the C++ functions of the ASORA
 * library. Care has to be taken mostly with the numpy array arguments, since the
 * underlying raw C pointer is passed directly to the C++ functions with little checks.
 */

namespace {

    /// Helper function to map C++ types to NPY_TYPES for type checking
    template <typename T>
    NPY_TYPES getNpyType();

    template <>
    NPY_TYPES getNpyType<double>() {
        return NPY_DOUBLE;
    }

    template <>
    NPY_TYPES getNpyType<int>() {
        return NPY_INT;
    }

    /// Perform type checking on numpy arrays.
    template <typename T>
    bool numpy_check(const PyArrayObject *array) {
        if (!PyArray_Check(array) || PyArray_TYPE(array) != getNpyType<T>()) {
            using namespace std::string_literals;
            std::string msg =
                "array must be a numpy NDArray of type "s + typeid(T).name();
            PyErr_SetString(PyExc_TypeError, msg.c_str());
            return false;
        }
        return true;
    }

    /// Load numpy array data to device buffer with error handling
    template <typename T>
    bool load_array_to_device(
        const PyArrayObject *array, asora::buffer_tag tag, unsigned int gpu_id = 0
    ) {
        if (!numpy_check<T>(array)) return false;

        auto data = static_cast<T *>(PyArray_DATA(array));
        auto size = static_cast<size_t>(PyArray_SIZE(array));

        try {
            asora::get_device_pool(gpu_id).transfer<T>(tag, data, size);
        } catch (const std::exception &e) {
            PyErr_SetString(PyExc_TypeError, e.what());
            return false;
        }
        return true;
    }

}  // namespace

/// Expose asora::do_all_sources
PyObject *asora_do_all_sources([[maybe_unused]] PyObject *self, PyObject *args) {
    double R;
    double sig;
    double dr;
    PyArrayObject *xh_av;
    PyArrayObject *phi_ion;
    size_t num_src;
    size_t m1;
    double minlogtau;
    double dlogtau;
    size_t num_tau;
    size_t grid_size;
    size_t block_size = 256;
    unsigned int gpu_id = 0;

    if (!PyArg_ParseTuple(
            args, "dddOOkkddkk|kI", &R, &sig, &dr, &xh_av, &phi_ion, &num_src, &m1,
            &minlogtau, &dlogtau, &num_tau, &grid_size, &block_size, &gpu_id
        ))
        return nullptr;

    // Error checking
    if (!numpy_check<double>(xh_av) || !numpy_check<double>(phi_ion)) return nullptr;

    // Get Array data
    auto xh_av_data = static_cast<double *>(PyArray_DATA(xh_av));
    auto phi_ion_data = static_cast<double *>(PyArray_DATA(phi_ion));

    try {
        asora::do_all_sources_gpu(
            R, sig, dr, xh_av_data, phi_ion_data, num_src, m1, minlogtau, dlogtau,
            num_tau, grid_size, block_size, gpu_id
        );
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }

    return Py_None;
}

/// Create a memory pool instance for the specified GPU.
PyObject *asora_device_init([[maybe_unused]] PyObject *self, PyObject *args) {
    unsigned int gpu_id = 0;
    if (!PyArg_ParseTuple(args, "|I", &gpu_id)) return nullptr;

    try {
        // Initialize the device
        asora::get_device_pool(gpu_id);
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_MemoryError, e.what());
        return nullptr;
    }

    return Py_None;
}

/// Expose asora::memory_pool::free
PyObject *asora_device_close([[maybe_unused]] PyObject *self, PyObject *args) {
    unsigned int gpu_id = 0;
    if (!PyArg_ParseTuple(args, "|I", &gpu_id)) return nullptr;

    try {
        asora::get_device_pool(gpu_id).free();
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_MemoryError, e.what());
        return nullptr;
    }
    return Py_None;
}

/// Allocate and copy density grid to the device.
PyObject *asora_density_to_device([[maybe_unused]] PyObject *self, PyObject *args) {
    PyArrayObject *ndens;
    unsigned int gpu_id = 0;
    if (!PyArg_ParseTuple(args, "O|I", &ndens, &gpu_id)) return nullptr;
    if (!load_array_to_device<double>(ndens, asora::buffer_tag::number_density, gpu_id))
        return nullptr;
    return Py_None;
}

/// Allocate and copy radiation tables to the device.
PyObject *asora_photo_table_to_device([[maybe_unused]] PyObject *self, PyObject *args) {
    PyArrayObject *thin_table, *thick_table;
    unsigned int gpu_id = 0;
    if (!PyArg_ParseTuple(args, "OO|I", &thin_table, &thick_table, &gpu_id))
        return nullptr;
    if (!load_array_to_device<double>(
            thin_table, asora::buffer_tag::photo_ion_thin_table, gpu_id
        ))
        return nullptr;
    if (!load_array_to_device<double>(
            thick_table, asora::buffer_tag::photo_ion_thick_table, gpu_id
        ))
        return nullptr;
    return Py_None;
}

/// Allocate and copy source properties to the device.
PyObject *asora_source_data_to_device([[maybe_unused]] PyObject *self, PyObject *args) {
    PyArrayObject *src_pos, *src_flux;
    unsigned int gpu_id = 0;
    if (!PyArg_ParseTuple(args, "OO|I", &src_pos, &src_flux, &gpu_id)) return nullptr;
    if (!load_array_to_device<int>(src_pos, asora::buffer_tag::source_position, gpu_id))
        return nullptr;
    if (!load_array_to_device<double>(src_flux, asora::buffer_tag::source_flux, gpu_id))
        return nullptr;
    return Py_None;
}

PyObject *asora_chemistry_global_pass([[maybe_unused]] PyObject *self, PyObject *args) {
    double dt;
    PyArrayObject *ndens;
    PyArrayObject *temp;
    PyArrayObject *xh;
    PyArrayObject *xh_av;
    PyArrayObject *xh_int;
    PyArrayObject *phi_ion;
    PyArrayObject *clump;
    double bh00;
    double albpow;
    double colh0;
    double temph0;
    double abu_c;
    size_t block_size = 512;

    if (!PyArg_ParseTuple(
            args, "dOOOOOOOddddd|k", &dt, &ndens, &temp, &xh, &xh_av, &xh_int, &phi_ion,
            &clump, &bh00, &albpow, &colh0, &temph0, &abu_c, &block_size
        ))
        return nullptr;

    // Get Array data
    auto xh_data = static_cast<double *>(PyArray_DATA(xh));
    auto xh_av_data = static_cast<double *>(PyArray_DATA(xh_av));
    auto xh_int_data = static_cast<double *>(PyArray_DATA(xh_int));
    auto temp_data = static_cast<double *>(PyArray_DATA(temp));
    auto ndens_data = static_cast<double *>(PyArray_DATA(ndens));
    auto phi_ion_data = static_cast<double *>(PyArray_DATA(phi_ion));
    auto clump_data = static_cast<double *>(PyArray_DATA(clump));
    auto n_cells = static_cast<size_t>(PyArray_SIZE(xh));

    try {
        auto conv_flag = asora::global_pass(
            xh_data, xh_av_data, xh_int_data, temp_data, ndens_data, phi_ion_data,
            clump_data, dt, bh00, albpow, colh0, temph0, abu_c, n_cells, block_size
        );
        return Py_BuildValue("k", conv_flag);
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
    return Py_None;
}

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

static PyMethodDef asoraMethods[] = {
    {"do_all_sources", asora_do_all_sources, METH_VARARGS, "Perform ASORA raytracing"},
    {"device_init", asora_device_init, METH_VARARGS,
     "Initialize device and allocate memory"},
    {"device_close", asora_device_close, METH_VARARGS, "Close device and free memory"},
    {"density_to_device", asora_density_to_device, METH_VARARGS,
     "Copy density field to the device"},
    {"photo_table_to_device", asora_photo_table_to_device, METH_VARARGS,
     "Copy radiation tables to the device"},
    {"source_data_to_device", asora_source_data_to_device, METH_VARARGS,
     "Copy source data to the device"},
    {"chemistry_global_pass", asora_chemistry_global_pass, METH_VARARGS,
     "Solve chemistry ODE"},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

static struct PyModuleDef asoramodule = {
    PyModuleDef_HEAD_INIT, "libasora",
    "CUDA C++ implementation of the short-characteristics RT", -1, asoraMethods
};

PyMODINIT_FUNC PyInit_libasora(void) {
    PyObject *module = PyModule_Create(&asoramodule);
    import_array();
    return module;
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
