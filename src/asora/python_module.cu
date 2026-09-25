#include <Python.h>

#include "chemistry.h"
#include "memory.h"
#include "raytracing.cuh"

#include <numpy/arrayobject.h>

/* @file python_module.cu
 * @brief ASORA Python C-extension module
 *
 * This file contains the wrappers for python to access the C++ functions of the ASORA
 * library. Care has to be taken mostly with the numpy array arguments, since the
 * underlying raw C pointer is passed directly to the C++ functions with little checks.
 */

#define SAFE_CHECK_INITIALIZED()                       \
    try {                                              \
        asora::device::check_initialized();            \
    } catch (const std::exception &e) {                \
        PyErr_SetString(PyExc_RuntimeError, e.what()); \
        return nullptr;                                \
    }

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
        const PyArrayObject *np_array, asora::device_array<T> &array
    ) {
        if (!numpy_check<T>(np_array)) return false;

        auto data = static_cast<T *>(PyArray_DATA(np_array));
        auto size = static_cast<size_t>(PyArray_SIZE(np_array));

        try {
            array.assign(data, size);
        } catch (const std::exception &e) {
            PyErr_SetString(PyExc_ValueError, e.what());
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
    PyArrayObject *phi_ion;
    size_t num_src;
    size_t m1;
    double minlogtau;
    double dlogtau;
    size_t num_tau;
    size_t grid_size;
    size_t block_size = 256;

    if (!PyArg_ParseTuple(
            args, "dddOkkddkk|k", &R, &sig, &dr, &phi_ion, &num_src, &m1, &minlogtau,
            &dlogtau, &num_tau, &grid_size, &block_size
        ))
        return nullptr;

    // Error checking
    if (!numpy_check<double>(phi_ion)) return nullptr;

    // Get Array data
    auto phi_ion_data = static_cast<double *>(PyArray_DATA(phi_ion));

    try {
        asora::do_all_sources_gpu(
            R, sig, dr, phi_ion_data, num_src, m1, minlogtau, dlogtau, num_tau,
            block_size
        );
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }

    return Py_None;
}

/// Expose asora::device::initialize
PyObject *asora_device_init([[maybe_unused]] PyObject *self, PyObject *args) {
    unsigned int mpi_rank = 0;
    if (!PyArg_ParseTuple(args, "|I", &mpi_rank)) return nullptr;

    try {
        asora::device::initialize(mpi_rank);
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_MemoryError, e.what());
        return nullptr;
    }

    return Py_None;
}

/// Expose asora::device::close
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

/// Expose asora::device::is_initialized.
PyObject *asora_is_device_init([[maybe_unused]] PyObject *self, PyObject *args) {
    if (!PyArg_ParseTuple(args, "")) return nullptr;

    return asora::device::is_initialized() ? Py_True : Py_False;
}

/// Expose whether the extension was compiled with periodic boundary mode.
PyObject *asora_is_periodic_mode_active(
    [[maybe_unused]] PyObject *self, [[maybe_unused]] PyObject *args
) {
    if (!PyArg_ParseTuple(args, "")) return nullptr;

#if defined(PERIODIC)
    Py_RETURN_TRUE;
#else
    Py_RETURN_FALSE;
#endif
}

/// Allocate and copy density grid to the device.
PyObject *asora_density_to_device([[maybe_unused]] PyObject *self, PyObject *args) {
    SAFE_CHECK_INITIALIZED();

    using namespace asora;
    namespace tag = resident_tag;
    PyArrayObject *ndens;
    return PyArg_ParseTuple(args, "O", &ndens) &&  //
                   load_array_to_device(ndens, device::resident<tag::number_density>())
               ? Py_None
               : nullptr;
}

/// Allocate and copy radiation tables to the device.
PyObject *asora_photo_table_to_device([[maybe_unused]] PyObject *self, PyObject *args) {
    SAFE_CHECK_INITIALIZED();

    using namespace asora;
    namespace tag = resident_tag;
    PyArrayObject *thin_table, *thick_table;
    return PyArg_ParseTuple(args, "OO", &thin_table, &thick_table) &&
                   load_array_to_device(
                       thin_table, device::resident<tag::photo_ion_thin>()
                   ) &&
                   load_array_to_device(
                       thick_table, device::resident<tag::photo_ion_thick>()
                   )
               ? Py_None
               : nullptr;
}

/// Allocate and copy source properties to the device.
PyObject *asora_source_data_to_device([[maybe_unused]] PyObject *self, PyObject *args) {
    SAFE_CHECK_INITIALIZED();

    using namespace asora;
    namespace tag = resident_tag;
    PyArrayObject *src_pos, *src_flux;
    return PyArg_ParseTuple(args, "OO", &src_pos, &src_flux) &&
                   load_array_to_device(
                       src_pos, device::resident<tag::source_position>()
                   ) &&
                   load_array_to_device(src_flux, device::resident<tag::source_flux>())
               ? Py_None
               : nullptr;
}

/// Allocate and copy the fields that stay constant for one timestep.
PyObject *asora_timestep_data_to_device(
    [[maybe_unused]] PyObject *self, PyObject *args
) {
    SAFE_CHECK_INITIALIZED();
    PyArrayObject *xh, *temp, *clump;
    if (!PyArg_ParseTuple(args, "OOO", &xh, &temp, &clump)) return nullptr;

    using namespace asora;
    namespace tag = resident_tag;
    if (!load_array_to_device(xh, device::resident<tag::fraction_HII>()) ||
        !load_array_to_device(temp, device::resident<tag::temperature>()) ||
        !load_array_to_device(clump, device::resident<tag::clumping>()))
        return nullptr;

    // The average fraction starts the timestep equal to the initial one, so it is
    // seeded device-to-device rather than uploaded a second time. From here on it
    // is updated in place by the chemistry pass and never leaves the device,
    // except where a rank has to broadcast it.
    try {
        device::resident<tag::fraction_HII_avg>().assign(
            device::resident<tag::fraction_HII>()
        );
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_ValueError, e.what());
        return nullptr;
    }
    return Py_None;
}

/// Copy the average ionized fraction to the device, for ranks that receive it.
PyObject *asora_average_fraction_to_device(
    [[maybe_unused]] PyObject *self, PyObject *args
) {
    SAFE_CHECK_INITIALIZED();

    using namespace asora;
    namespace tag = resident_tag;
    PyArrayObject *xh_av;
    return PyArg_ParseTuple(
               args, "O", &xh_av
           ) && load_array_to_device(xh_av, device::resident<tag::fraction_HII_avg>())
               ? Py_None
               : nullptr;
}

/// Copy the average ionized fraction back to the host, for ranks that broadcast it.
PyObject *asora_average_fraction_to_host(
    [[maybe_unused]] PyObject *self, PyObject *args
) {
    SAFE_CHECK_INITIALIZED();

    PyArrayObject *xh_av;
    if (!PyArg_ParseTuple(args, "O", &xh_av)) return nullptr;
    if (!numpy_check<double>(xh_av)) return nullptr;

    using namespace asora;
    namespace tag = resident_tag;
    const auto &frac_HII_avg = device::resident<tag::fraction_HII_avg>();
    if (!frac_HII_avg) {
        PyErr_SetString(
            PyExc_RuntimeError,
            "average ionized fraction is not allocated on the device; call "
            "timestep_data_to_device first"
        );
        return nullptr;
    }

    try {
        frac_HII_avg.copy_to_host(
            static_cast<double *>(PyArray_DATA(xh_av)),
            static_cast<size_t>(PyArray_SIZE(xh_av))
        );
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_ValueError, e.what());
        return nullptr;
    }
    return Py_None;
}

PyObject *asora_chemistry_global_pass([[maybe_unused]] PyObject *self, PyObject *args) {
    double dt;
    PyArrayObject *xh_int;
    PyArrayObject *phi_ion;
    double bh00;
    double albpow;
    double colh0;
    double temph0;
    double abu_c;
    size_t block_size = 512;

    if (!PyArg_ParseTuple(
            args, "dOOddddd|k", &dt, &xh_int, &phi_ion, &bh00, &albpow, &colh0, &temph0,
            &abu_c, &block_size
        ))
        return nullptr;

    if (!numpy_check<double>(xh_int) || !numpy_check<double>(phi_ion)) return nullptr;

    // Get Array data
    auto xh_int_data = static_cast<double *>(PyArray_DATA(xh_int));
    auto phi_ion_data = static_cast<double *>(PyArray_DATA(phi_ion));
    auto n_cells = static_cast<size_t>(PyArray_SIZE(xh_int));

    try {
        auto conv_flag = asora::global_pass(
            xh_int_data, phi_ion_data, dt, bh00, albpow, colh0, temph0, abu_c, n_cells,
            block_size
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
    {"is_device_init", asora_is_device_init, METH_VARARGS,
     "Check if the device is initialized"},
    {"is_periodic_mode_active", asora_is_periodic_mode_active, METH_VARARGS,
     "Check if libasora was compiled with PERIODIC"},
    {"density_to_device", asora_density_to_device, METH_VARARGS,
     "Copy density field to the device"},
    {"photo_table_to_device", asora_photo_table_to_device, METH_VARARGS,
     "Copy radiation tables to the device"},
    {"source_data_to_device", asora_source_data_to_device, METH_VARARGS,
     "Copy source data to the device"},
    {"timestep_data_to_device", asora_timestep_data_to_device, METH_VARARGS,
     "Copy the fields that stay constant over a timestep to the device"},
    {"average_fraction_to_device", asora_average_fraction_to_device, METH_VARARGS,
     "Copy the average ionized fraction to the device"},
    {"average_fraction_to_host", asora_average_fraction_to_host, METH_VARARGS,
     "Copy the average ionized fraction back to the host"},
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
