#include <Python.h>

#include "chemistry.h"
#include "memory.h"
#include "octahedron.cuh"
#include "raytracing.cuh"
#include "shortchar.cuh"
#include "utils.cuh"

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
    PyObject *build_lut_entry(const asora::shortchar_entry &entry) {
        PyObject *obj = PyStructSequence_New(LutEntryType);
        if (!obj) return nullptr;

        PyObject *indices = Py_BuildValue(
            "IIII", entry.info.indices[0], entry.info.indices[1], entry.info.indices[2],
            entry.info.indices[3]
        );
        if (!indices) {
            Py_DECREF(obj);
            return nullptr;
        }

        PyStructSequence_SetItem(obj, 0, PyLong_FromLong(entry.pos.x));
        PyStructSequence_SetItem(obj, 1, PyLong_FromLong(entry.pos.y));
        PyStructSequence_SetItem(obj, 2, PyLong_FromLong(entry.pos.z));
        PyStructSequence_SetItem(obj, 3, PyFloat_FromDouble(entry.info.dx));
        PyStructSequence_SetItem(obj, 4, PyFloat_FromDouble(entry.info.dy));
        PyStructSequence_SetItem(obj, 5, PyFloat_FromDouble(entry.info.path));
        PyStructSequence_SetItem(obj, 6, indices);

        return obj;
    }

    PyObject *create_lut_list(const asora::shortchar_entries &lut) {
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

PyObject *asora_shell2cart([[maybe_unused]] PyObject *self, PyObject *args) {
    int q, s;
    if (!PyArg_ParseTuple(args, "ii", &q, &s)) return nullptr;

    try {
        auto [i, j, k] = asora::shell2cart(q, s);
        return Py_BuildValue("iii", i, j, k);
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_MemoryError, e.what());
        return nullptr;
    }
}

PyObject *asora_cart2shell([[maybe_unused]] PyObject *self, PyObject *args) {
    int3 pos;
    if (!PyArg_ParseTuple(args, "iii", &pos.x, &pos.y, &pos.z)) return nullptr;

    try {
        auto [q, s] = asora::cart2shell(pos);
        return Py_BuildValue("ii", q, s);
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_MemoryError, e.what());
        return nullptr;
    }
}

PyObject *asora_index2shell([[maybe_unused]] PyObject *self, PyObject *args) {
    int index;
    if (!PyArg_ParseTuple(args, "i", &index)) return nullptr;

    if (index < 0) {
        PyErr_SetString(PyExc_ValueError, "index must be non-negative");
        return nullptr;
    }

    auto [q, s] = asora::index2shell(static_cast<size_t>(index));
    return Py_BuildValue("ii", q, s);
}

PyObject *asora_create_shortchar_lut([[maybe_unused]] PyObject *self, PyObject *args) {
    double R = 0.0;
    int N = 0;
    if (!PyArg_ParseTuple(args, "di", &R, &N)) return nullptr;

    if (N <= 0) {
        PyErr_SetString(PyExc_ValueError, "N must be positive");
        return nullptr;
    }

    // Errors are reported by an unset LUT, not by exceptions.
    auto lut = asora::create_shortchar_interp_lut(R, static_cast<size_t>(N));
    if (!lut.is_set()) {
        PyErr_SetString(
            PyExc_RuntimeError, "Raytracing lookup table could not be built"
        );
        return nullptr;
    }

    return Py_BuildValue("i", lut.q_max());
}

PyObject *asora_get_shortchar_lut([[maybe_unused]] PyObject *self, PyObject *args) {
    if (!PyArg_ParseTuple(args, "")) return nullptr;

    asora::shortchar_entries lut;
    try {
        lut = asora::copy_shortchar_lut_to_host();
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

PyObject *asora_compute_shortchar_factors(
    [[maybe_unused]] PyObject *self, PyObject *args
) {
    int di, dj, dk;
    if (!PyArg_ParseTuple(args, "iii", &di, &dj, &dk)) return nullptr;

    float3 factors;
    try {
        factors = asora::compute_shortchar_factors(di, dj, dk);
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }

    return Py_BuildValue("fff", factors.x, factors.y, factors.z);
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

    Py_RETURN_NONE;
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
    Py_RETURN_NONE;
}

/// Expose asora::device::trim.
PyObject *asora_device_trim([[maybe_unused]] PyObject *self, PyObject *args) {
    SAFE_CHECK_INITIALIZED();

    if (!PyArg_ParseTuple(args, "")) return nullptr;

    try {
        asora::device::trim();
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
    Py_RETURN_NONE;
}

/// Expose asora::device::is_initialized.
PyObject *asora_is_device_init([[maybe_unused]] PyObject *self, PyObject *args) {
    if (!PyArg_ParseTuple(args, "")) return nullptr;

    return Py_NewRef(asora::device::is_initialized() ? Py_True : Py_False);
}

PyObject *asora_in_domain([[maybe_unused]] PyObject *self, PyObject *args) {
    int3 pos, off;
    size_t N;
    if (!PyArg_ParseTuple(
            args, "(iii)(iii)k", &pos.x, &pos.y, &pos.z, &off.x, &off.y, &off.z, &N
        ))
        return nullptr;

    return Py_NewRef(asora::in_domain(pos, off, N) ? Py_True : Py_False);
}

/// Expose whether the extension was compiled with periodic boundary mode.
PyObject *asora_is_periodic_mode_active(
    [[maybe_unused]] PyObject *self, [[maybe_unused]] PyObject *args
) {
    if (!PyArg_ParseTuple(args, "")) return nullptr;

    if constexpr (asora::periodic_conditions)
        Py_RETURN_TRUE;
    else
        Py_RETURN_FALSE;
}

/// Expose the number of q-shells kept in the column density ring buffer.
PyObject *asora_num_column_density_banks(
    [[maybe_unused]] PyObject *self, [[maybe_unused]] PyObject *args
) {
    if (!PyArg_ParseTuple(args, "")) return nullptr;

    return PyLong_FromSize_t(asora::num_banks);
}

/// Allocate and copy density grid to the device.
PyObject *asora_density_to_device([[maybe_unused]] PyObject *self, PyObject *args) {
    SAFE_CHECK_INITIALIZED();

    PyArrayObject *ndens;
    if (!PyArg_ParseTuple(args, "O", &ndens)) return nullptr;

    using namespace asora;
    namespace tag = resident_tag;
    if (!load_array_to_device(ndens, device::resident<tag::number_density>()))
        return nullptr;

    Py_RETURN_NONE;
}

/// Allocate and copy radiation tables to the device.
PyObject *asora_photo_table_to_device([[maybe_unused]] PyObject *self, PyObject *args) {
    SAFE_CHECK_INITIALIZED();

    PyArrayObject *thin_table, *thick_table;
    if (!PyArg_ParseTuple(args, "OO", &thin_table, &thick_table)) return nullptr;

    using namespace asora;
    namespace tag = resident_tag;
    if (!load_array_to_device(thin_table, device::resident<tag::photo_ion_thin>()) ||
        !load_array_to_device(thick_table, device::resident<tag::photo_ion_thick>()))
        return nullptr;

    Py_RETURN_NONE;
}

/// Allocate and copy source properties to the device.
PyObject *asora_source_data_to_device([[maybe_unused]] PyObject *self, PyObject *args) {
    SAFE_CHECK_INITIALIZED();

    PyArrayObject *src_pos, *src_flux;
    if (!PyArg_ParseTuple(args, "OO", &src_pos, &src_flux)) return nullptr;

    using namespace asora;
    namespace tag = resident_tag;
    if (!load_array_to_device(src_pos, device::resident<tag::source_position>()) ||
        !load_array_to_device(src_flux, device::resident<tag::source_flux>()))
        return nullptr;

    Py_RETURN_NONE;
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

    Py_RETURN_NONE;
}

/// Copy the average ionized fraction to the device, for ranks that receive it.
PyObject *asora_average_fraction_to_device(
    [[maybe_unused]] PyObject *self, PyObject *args
) {
    SAFE_CHECK_INITIALIZED();
    PyArrayObject *xh_av;
    if (!PyArg_ParseTuple(args, "O", &xh_av)) return nullptr;

    using namespace asora;
    namespace tag = resident_tag;
    if (!load_array_to_device(xh_av, device::resident<tag::fraction_HII_avg>()))
        return nullptr;

    Py_RETURN_NONE;
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
            "average ionized fraction is not allocated on the device"
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

    Py_RETURN_NONE;
}

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
            grid_size, block_size
        );
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_ValueError, e.what());
        return nullptr;
    }

    Py_RETURN_NONE;
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
}

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

static PyMethodDef asora_methods[] = {
    {"cells_in_shell", asora_cells_in_shell, METH_VARARGS,
     "Number of cells in q-shell"},
    {"cells_to_shell", asora_cells_to_shell, METH_VARARGS,
     "Cumulative number of cells up to q-shell"},
    {"shell2cart", asora_shell2cart, METH_VARARGS,
     "Convert shell indexing to cartesian coordinates"},
    {"cart2shell", asora_cart2shell, METH_VARARGS,
     "Convert cartesian coordinates to shell indexing"},
    {"index2shell", asora_index2shell, METH_VARARGS,
     "Convert a cumulative cell index to shell indexing"},
    {"create_shortchar_lut", asora_create_shortchar_lut, METH_VARARGS,
     "Create LUT for ASORA short-characteristic interpolation; returns q_max"},
    {"get_shortchar_lut", asora_get_shortchar_lut, METH_VARARGS,
     "Create and copy short-characteristic interpolation LUT to Python"},
    {"pack_offset", asora_pack_offset, METH_VARARGS,
     "Pack cell offset into a single integer"},
    {"unpack_offset", asora_unpack_offset, METH_VARARGS,
     "Unpack cell offset from a single integer"},
    {"compute_shortchar_factors", asora_compute_shortchar_factors, METH_VARARGS,
     "Compute geometric factors for short-characteristic interpolation"},
    {"device_init", asora_device_init, METH_VARARGS,
     "Initialize device and allocate memory"},
    {"device_close", asora_device_close, METH_VARARGS, "Close device and free memory"},
    {"device_trim", asora_device_trim, METH_VARARGS,
     "Return the memory cached by the scratch pool to the device"},
    {"is_device_init", asora_is_device_init, METH_VARARGS,
     "Check if the device is initialized"},
    {"in_domain", asora_in_domain, METH_VARARGS,
     "Check if a cell offset from a source is within the domain"},
    {"is_periodic_mode_active", asora_is_periodic_mode_active, METH_VARARGS,
     "Check if libasora was compiled with PERIODIC"},
    {"num_column_density_banks", asora_num_column_density_banks, METH_VARARGS,
     "Number of q-shells kept in the column density ring buffer"},
    {"density_to_device", asora_density_to_device, METH_VARARGS,
     "Copy density field to the device"},
    {"photo_table_to_device", asora_photo_table_to_device, METH_VARARGS,
     "Copy radiation tables to the device"},
    {"source_data_to_device", asora_source_data_to_device, METH_VARARGS,
     "Copy source data to the device"},
    {"do_all_sources", asora_do_all_sources, METH_VARARGS, "Perform ASORA raytracing"},
    {"timestep_data_to_device", asora_timestep_data_to_device, METH_VARARGS,
     "Copy the fields that stay constant over a timestep to the device"},
    {"average_fraction_to_device", asora_average_fraction_to_device, METH_VARARGS,
     "Copy the average ionized fraction to the device"},
    {"average_fraction_to_host", asora_average_fraction_to_host, METH_VARARGS,
     "Copy the average ionized fraction back to the host"},
    {"chemistry_global_pass", asora_chemistry_global_pass, METH_VARARGS,
     "Solve chemistry ODE"},
    {NULL, NULL, 0, NULL}  // sentinel
};

static struct PyModuleDef asoramodule = {
    PyModuleDef_HEAD_INIT,
    "libasora",                                                 // m_name
    "CUDA C++ implementation of the short-characteristics RT",  // m_doc
    -1,                                                         // m_size
    asora_methods,                                              // m_methods
    nullptr,                                                    // m_slots
    nullptr,                                                    // m_traverse
    nullptr,                                                    // m_clear
    nullptr                                                     // m_free
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
