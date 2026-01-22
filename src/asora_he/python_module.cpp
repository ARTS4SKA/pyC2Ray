#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_SSIZE_T_CLEAN

#include "../asora/memory.h"
#include "../asora/utils.cuh"
#include "raytracing.cuh"

#include <Python.h>
#include <numpy/arrayobject.h>

#include <string>
#include <typeinfo>

namespace {

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

    template <typename T>
    bool load_array_to_device(const PyArrayObject *array, asora::buffer_tag tag) {
        if (!numpy_check<T>(array)) return false;

        auto data = static_cast<T *>(PyArray_DATA(array));
        auto size = static_cast<size_t>(PyArray_SIZE(array));

        try {
            asora::device::transfer<T>(tag, data, size);
        } catch (const std::exception &e) {
            PyErr_SetString(PyExc_TypeError, e.what());
            return false;
        }
        return true;
    }

}  // namespace

// ===========================================================================
// ASORA Python C-extension module
// Mostly boilerplate code, this file contains the wrappers for python
// to access the C++ functions of the ASORA library. Care has to be taken
// mostly with the numpy array arguments, since the underlying raw C pointer
// is passed directly to the C++ functions without additional type checking.
// ===========================================================================

// ========================================================================
// Raytrace all sources and compute photoionization rates
// ========================================================================
static PyObject *asora_do_all_sources(PyObject *self, PyObject *args) {
    double R;
    PyArrayObject *sig_HI;
    PyArrayObject *sig_HeI;
    PyArrayObject *sig_HeII;
    int nbin1;
    int nbin2;
    int nbin3;
    int num_freq;
    double dr;
    PyArrayObject *xHII_av;
    PyArrayObject *xHeII_av;
    PyArrayObject *xHeIII_av;
    PyArrayObject *phion_HI;
    PyArrayObject *phion_HeI;
    PyArrayObject *phion_HeII;
    PyArrayObject *pheat_HI;
    PyArrayObject *pheat_HeI;
    PyArrayObject *pheat_HeII;
    int num_src;
    int m1;
    double minlogtau;
    double dlogtau;
    int num_tau;
    size_t grid_size;
    size_t block_size = 256;

    if (!PyArg_ParseTuple(
            args, "dOOOiiiidOOOOOOOOOiiddik|k", &R, &sig_HI, &sig_HeI, &sig_HeII,
            &nbin1, &nbin2, &nbin3, &num_freq, &dr, &xHII_av, &xHeII_av, &xHeIII_av,
            &phion_HI, &phion_HeI, &phion_HeII, &pheat_HI, &pheat_HeI, &pheat_HeII,
            &num_src, &m1, &minlogtau, &dlogtau, &num_tau, &grid_size, &block_size
        ))
        return NULL;

    // Type checking
    if (!numpy_check<double>(sig_HI) || !numpy_check<double>(sig_HeI) ||
        !numpy_check<double>(sig_HeII) || !numpy_check<double>(xHII_av) ||
        !numpy_check<double>(xHeII_av) || !numpy_check<double>(xHeIII_av) ||
        !numpy_check<double>(phion_HI) || !numpy_check<double>(phion_HeI) ||
        !numpy_check<double>(phion_HeII) || !numpy_check<double>(pheat_HI) ||
        !numpy_check<double>(pheat_HeI) || !numpy_check<double>(pheat_HeII))
        return nullptr;

    // Get Array data
    auto sig_HI_data = static_cast<double *>(PyArray_DATA(sig_HI));
    auto sig_HeI_data = static_cast<double *>(PyArray_DATA(sig_HeI));
    auto sig_HeII_data = static_cast<double *>(PyArray_DATA(sig_HeII));
    auto phion_HI_data = static_cast<double *>(PyArray_DATA(phion_HI));
    auto phion_HeI_data = static_cast<double *>(PyArray_DATA(phion_HeI));
    auto phion_HeII_data = static_cast<double *>(PyArray_DATA(phion_HeII));
    auto pheat_HI_data = static_cast<double *>(PyArray_DATA(pheat_HI));
    auto pheat_HeI_data = static_cast<double *>(PyArray_DATA(pheat_HeI));
    auto pheat_HeII_data = static_cast<double *>(PyArray_DATA(pheat_HeII));
    auto xh_av_HI_data = static_cast<double *>(PyArray_DATA(xHII_av));
    auto xh_av_HeI_data = static_cast<double *>(PyArray_DATA(xHeII_av));
    auto xh_av_HeII_data = static_cast<double *>(PyArray_DATA(xHeIII_av));

    try {
        asora::do_all_sources_gpu(
            R, sig_HI_data, sig_HeI_data, sig_HeII_data, nbin1, nbin2, nbin3, num_freq,
            dr, xh_av_HI_data, xh_av_HeI_data, xh_av_HeII_data, phion_HI_data,
            phion_HeI_data, phion_HeII_data, pheat_HI_data, pheat_HeI_data,
            pheat_HeII_data, num_src, m1, minlogtau, dlogtau, num_tau, grid_size,
            block_size
        );
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }

    return Py_None;
}

// Initialize GPU device and allocate some memory for grid data
PyObject *asora_device_init([[maybe_unused]] PyObject *self, PyObject *args) {
    unsigned int mpi_rank = 0;
    if (!PyArg_ParseTuple(args, "|I", &mpi_rank)) return nullptr;

    try {
        // Initialize the device
        asora::device::initialize(mpi_rank);
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_MemoryError, e.what());
        return nullptr;
    }

    return Py_None;
}

// Close device and deallocate memory
PyObject *asora_device_close([[maybe_unused]] PyObject *self, PyObject *args) {
    if (!PyArg_ParseTuple(args, "")) return nullptr;

    try {
        asora::device::close();
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_MemoryError, e.what());
        return nullptr;
    }
    return Py_None;
}

PyObject *asora_is_device_init([[maybe_unused]] PyObject *self, PyObject *args) {
    if (!PyArg_ParseTuple(args, "")) return nullptr;

    return asora::device::is_initialized() ? Py_True : Py_False;
}

// Copy density grid to GPU
PyObject *asora_density_to_device([[maybe_unused]] PyObject *self, PyObject *args) {
    PyArrayObject *ndens;
    return PyArg_ParseTuple(args, "O", &ndens) &&  //
                   load_array_to_device<double>(
                       ndens, asora::buffer_tag::number_density
                   )
               ? Py_None
               : nullptr;
}

// Copy radiation tables to GPU
PyObject *asora_tables_to_device([[maybe_unused]] PyObject *self, PyObject *args) {
    PyArrayObject *phion_thin_table;
    PyArrayObject *phion_thick_table;
    PyArrayObject *pheat_thin_table;
    PyArrayObject *pheat_thick_table;
    if (!PyArg_ParseTuple(
            args, "OOOO", &phion_thin_table, &phion_thick_table, &pheat_thin_table,
            &pheat_thick_table
        ))
        return nullptr;

    using namespace asora;
    if (!load_array_to_device<double>(
            phion_thin_table, buffer_tag::photo_ion_thin_table
        ))
        return nullptr;
    if (!load_array_to_device<double>(
            phion_thick_table, buffer_tag::photo_ion_thick_table
        ))
        return nullptr;
    if (!load_array_to_device<double>(
            pheat_thin_table, buffer_tag::photo_heat_thin_table
        ))
        return nullptr;
    if (!load_array_to_device<double>(
            pheat_thick_table, buffer_tag::photo_heat_thick_table
        ))
        return nullptr;
    return Py_None;
}

// Copy source data to GPU
PyObject *asora_source_data_to_device([[maybe_unused]] PyObject *self, PyObject *args) {
    PyArrayObject *src_pos, *src_flux;
    return PyArg_ParseTuple(args, "OO", &src_pos, &src_flux) &&
                   load_array_to_device<int>(
                       src_pos, asora::buffer_tag::source_position
                   ) &&
                   load_array_to_device<double>(
                       src_flux, asora::buffer_tag::source_flux
                   )
               ? Py_None
               : nullptr;
}

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// ========================================================================
// Define module functions and initialization function
// ========================================================================
static PyMethodDef asoraMethods[] = {
    {"do_all_sources", asora_do_all_sources, METH_VARARGS, "Do OCTA raytracing (GPU)"},
    {"device_init", asora_device_init, METH_VARARGS,
     "Initialize device and allocate memory"},
    {"device_close", asora_device_close, METH_VARARGS, "Close device and free memory"},
    {"is_device_init", asora_is_device_init, METH_VARARGS,
     "Check if the device is initialized"},
    {"density_to_device", asora_density_to_device, METH_VARARGS,
     "Copy density field to GPU"},
    {"tables_to_device", asora_tables_to_device, METH_VARARGS,
     "Copy radiation tables to GPU"},
    {"source_data_to_device", asora_source_data_to_device, METH_VARARGS,
     "Copy source data to GPU"},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

static struct PyModuleDef asoramodule = {
    PyModuleDef_HEAD_INIT, "libasora_He",
    "CUDA C++ implementation of the short-characteristics RT", -1, asoraMethods
};

PyMODINIT_FUNC PyInit_libasora_He(void) {
    PyObject *module = PyModule_Create(&asoramodule);
    import_array();
    return module;
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
