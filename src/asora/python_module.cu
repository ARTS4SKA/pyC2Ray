
#include "chemistry.h"
#include "memory.h"
#include "raytracing.cuh"
#include "raytracing_lut.cuh"
#include "utils.cuh"

#include <Python.h>
#include <numpy/arrayobject.h>

/* @file python_module.cu
 * @brief ASORA Python C-extension module
 *
 * This file contains the wrappers for python to access the C++ functions of the ASORA
 * library. Care has to be taken mostly with the numpy array arguments, since the
 * underlying raw C pointer is passed directly to the C++ functions with little checks.
 */

namespace {

    PyTypeObject *LutEntryType = nullptr;

    PyStructSequence_Field lut_entry_fields[] = {
        {"di", "cell offset in the i direction"},
        {"dj", "cell offset in the j direction"},
        {"dk", "cell offset in the k direction"},
        {"dx", "geometric factor along x"},
        {"dy", "geometric factor along y"},
        {"path", "path length through the cell"},
        {"indices", "tuple of 4 LUT indices of the interpolation neighbours"},
        {nullptr, nullptr}
    };

    PyStructSequence_Desc lut_entry_desc = {
        "libasora.LutEntry", "Look-up table entry produced by asora::create_lut",
        lut_entry_fields, 7
    };

    /// Convert a asora lut_entry to a LutEntry python object.
    PyObject *build_lut_entry(const asora::raytracing_lut::entry &item) {
        PyObject *obj = PyStructSequence_New(LutEntryType);
        if (!obj) return nullptr;

        PyObject *indices = Py_BuildValue(
            "kkkk", item.indices[0], item.indices[1], item.indices[2], item.indices[3]
        );
        if (!indices) {
            Py_DECREF(obj);
            return nullptr;
        }

        PyStructSequence_SetItem(obj, 0, PyLong_FromLong(item.di));
        PyStructSequence_SetItem(obj, 1, PyLong_FromLong(item.dj));
        PyStructSequence_SetItem(obj, 2, PyLong_FromLong(item.dk));
        PyStructSequence_SetItem(obj, 3, PyFloat_FromDouble(item.dx));
        PyStructSequence_SetItem(obj, 4, PyFloat_FromDouble(item.dy));
        PyStructSequence_SetItem(obj, 5, PyFloat_FromDouble(item.path));
        PyStructSequence_SetItem(obj, 6, indices);

        return obj;
    }

    PyObject *create_lut_list(const asora::raytracing_lut_entries &lut) {
        PyObject *result = PyList_New(static_cast<Py_ssize_t>(lut.size()));
        if (!result) return nullptr;

        for (size_t i = 0; i < lut.size(); ++i) {
            PyObject *entry = build_lut_entry(lut[i]);
            if (!entry) {
                Py_DECREF(result);
                return nullptr;
            }
            PyList_SET_ITEM(result, static_cast<Py_ssize_t>(i), entry);
        }

        return result;
    }

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
    bool load_array_to_device(const PyArrayObject *array, asora::buffer_tag tag) {
        if (!numpy_check<T>(array)) return false;

        auto data = static_cast<T *>(PyArray_DATA(array));
        auto size = static_cast<size_t>(PyArray_SIZE(array));

        try {
            asora::device::ensure_transfer<T>(tag, data, size);
        } catch (const std::exception &e) {
            PyErr_SetString(PyExc_ValueError, e.what());
            return false;
        }
        return true;
    }

}  // namespace

PyObject *asora_cells_in_shell([[maybe_unused]] PyObject *self, PyObject *args) {
    int q;
    if (!PyArg_ParseTuple(args, "i", &q)) return nullptr;

    auto n = asora::cells_in_shell(q);
    return PyLong_FromSize_t(n);
}

PyObject *asora_cells_to_shell([[maybe_unused]] PyObject *self, PyObject *args) {
    int q;
    if (!PyArg_ParseTuple(args, "i", &q)) return nullptr;

    auto n = asora::cells_to_shell(q);
    return PyLong_FromSize_t(n);
}

PyObject *asora_create_raytracing_lut([[maybe_unused]] PyObject *self, PyObject *args) {
    int q_max = 0;
    if (!PyArg_ParseTuple(args, "i", &q_max)) return nullptr;

    size_t n_cells = 0;
    try {
        // Initialize the device
        n_cells = asora::create_raytracing_lut(q_max);
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }

    return Py_BuildValue("k", n_cells);
}

PyObject *asora_get_raytracing_lut([[maybe_unused]] PyObject *self, PyObject *args) {
    int q_max = 0;
    if (!PyArg_ParseTuple(args, "i", &q_max)) return nullptr;

    asora::raytracing_lut_entries lut;
    try {
        asora::create_raytracing_lut(q_max);
        lut = asora::copy_lut_to_host(q_max);
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }

    return create_lut_list(lut);
}

PyObject *asora_pack_offset([[maybe_unused]] PyObject *self, PyObject *args) {
    int3 pos;
    if (!PyArg_ParseTuple(args, "iii", &pos.x, &pos.y, &pos.z)) return nullptr;

    uint32_t offset = 0;
    try {
        offset = asora::pack_offset(pos);
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }

    return PyLong_FromUnsignedLong(offset);
}

PyObject *asora_unpack_offset([[maybe_unused]] PyObject *self, PyObject *args) {
    unsigned int offset = 0;
    if (!PyArg_ParseTuple(args, "I", &offset)) return nullptr;

    int3 pos;
    try {
        pos = asora::unpack_offset(static_cast<uint32_t>(offset));
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }

    return Py_BuildValue("iii", pos.x, pos.y, pos.z);
}

/// Expose asora::device::initialize
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
    PyArrayObject *ndens;
    return PyArg_ParseTuple(args, "O", &ndens) &&  //
                   load_array_to_device<double>(
                       ndens, asora::buffer_tag::number_density
                   )
               ? Py_None
               : nullptr;
}

/// Allocate and copy radiation tables to the device.
PyObject *asora_photo_table_to_device([[maybe_unused]] PyObject *self, PyObject *args) {
    PyArrayObject *thin_table, *thick_table;
    return PyArg_ParseTuple(args, "OO", &thin_table, &thick_table) &&
                   load_array_to_device<double>(
                       thin_table, asora::buffer_tag::photo_ion_thin_table
                   ) &&
                   load_array_to_device<double>(
                       thick_table, asora::buffer_tag::photo_ion_thick_table
                   )
               ? Py_None
               : nullptr;
}

/// Allocate and copy source properties to the device.
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

/// Ensure mesh-dependent buffers exist with the expected size.
PyObject *asora_prepare_grid_buffers([[maybe_unused]] PyObject *self, PyObject *args) {
    size_t m1;
    int force_matching_size = 0;
    if (!PyArg_ParseTuple(args, "k|p", &m1, &force_matching_size)) return nullptr;

    auto n_cells = m1 * m1 * m1;
    try {
        asora::device::ensure<double>(
            asora::buffer_tag::number_density, n_cells,
            static_cast<bool>(force_matching_size)
        );
        asora::device::ensure<double>(
            asora::buffer_tag::fraction_HII, n_cells,
            static_cast<bool>(force_matching_size)
        );
        asora::device::ensure<double>(
            asora::buffer_tag::photo_ionization_HI, n_cells,
            static_cast<bool>(force_matching_size)
        );
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
    return Py_None;
}

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

    if (!PyArg_ParseTuple(
            args, "dddOOkkddkk|k", &R, &sig, &dr, &xh_av, &phi_ion, &num_src, &m1,
            &minlogtau, &dlogtau, &num_tau, &grid_size, &block_size
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
            num_tau, grid_size, block_size
        );
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }

    return Py_None;
}

PyObject *asora_chemistry_global_pass([[maybe_unused]] PyObject *self, PyObject *args) {
    double dt;
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
            args, "dOOOOOOddddd|k", &dt, &temp, &xh, &xh_av, &xh_int, &phi_ion, &clump,
            &bh00, &albpow, &colh0, &temph0, &abu_c, &block_size
        ))
        return nullptr;

    // Get Array data
    auto xh_data = static_cast<double *>(PyArray_DATA(xh));
    auto xh_av_data = static_cast<double *>(PyArray_DATA(xh_av));
    auto xh_int_data = static_cast<double *>(PyArray_DATA(xh_int));
    auto temp_data = static_cast<double *>(PyArray_DATA(temp));
    auto phi_ion_data = static_cast<double *>(PyArray_DATA(phi_ion));
    auto clump_data = static_cast<double *>(PyArray_DATA(clump));
    auto n_cells = static_cast<size_t>(PyArray_SIZE(xh));

    try {
        auto conv_flag = asora::global_pass(
            xh_data, xh_av_data, xh_int_data, temp_data, phi_ion_data, clump_data, dt,
            bh00, albpow, colh0, temph0, abu_c, n_cells, block_size
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
    {"cells_in_shell", asora_cells_in_shell, METH_VARARGS,
     "Number of cells in q-shell"},
    {"cells_to_shell", asora_cells_to_shell, METH_VARARGS,
     "Cumulative number of cells up to q-shell"},
    {"create_raytracing_lut", asora_create_raytracing_lut, METH_VARARGS,
     "Create LUT for ASORA raytracing"},
    {"get_raytracing_lut", asora_get_raytracing_lut, METH_VARARGS,
     "Create and copy look-up table for raytracing"},
    {"pack_offset", asora_pack_offset, METH_VARARGS,
     "Pack cell offset into a single integer"},
    {"unpack_offset", asora_unpack_offset, METH_VARARGS,
     "Unpack cell offset from a single integer"},
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
    {"prepare_grid_buffers", asora_prepare_grid_buffers, METH_VARARGS,
     "Ensure grid buffers are allocated with exact size for mesh m1"},
    {"do_all_sources", asora_do_all_sources, METH_VARARGS, "Perform ASORA raytracing"},
    {"chemistry_global_pass", asora_chemistry_global_pass, METH_VARARGS,
     "Solve chemistry ODE"},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

static struct PyModuleDef asoramodule = {
    PyModuleDef_HEAD_INIT, "libasora",
    "CUDA C++ implementation of the short-characteristics RT", -1, asoraMethods
};

PyMODINIT_FUNC PyInit_libasora(void) {
    PyObject *mod = PyModule_Create(&asoramodule);
    if (!mod) return nullptr;
    import_array();

    if (!LutEntryType) {
        LutEntryType = PyStructSequence_NewType(&lut_entry_desc);
        if (!LutEntryType) {
            Py_DECREF(mod);
            return nullptr;
        }
    }

    Py_INCREF(LutEntryType);
    if (PyModule_AddObject(
            mod, "LutEntry", reinterpret_cast<PyObject *>(LutEntryType)
        ) < 0) {
        Py_DECREF(LutEntryType);
        Py_DECREF(mod);
        return nullptr;
    }

    return mod;
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
