#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_SSIZE_T_CLEAN

#include "lut.h"
#include "tests.cuh"
#include "utils.cuh"

#include <Python.h>
#include <numpy/arrayobject.h>

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
        "libasoratest.LutEntry", "Look-up table entry produced by asora::create_lut",
        lut_entry_fields, 7
    };

    /// Convert an asora::lut_entry to a LutEntry python object.
    PyObject *build_lut_entry(const asora::lut_entry &item) {
        PyObject *obj = PyStructSequence_New(LutEntryType);
        if (!obj) return nullptr;

        PyObject *indices = Py_BuildValue(
            "kkkk", item.indices[0], item.indices[1], item.indices[2], item.indices[3]
        );
        if (!indices) {
            Py_DECREF(obj);
            return nullptr;
        }

        auto &&[di, dj, dk] = item.dijk();
        PyStructSequence_SetItem(obj, 0, PyLong_FromLong(di));
        PyStructSequence_SetItem(obj, 1, PyLong_FromLong(dj));
        PyStructSequence_SetItem(obj, 2, PyLong_FromLong(dk));
        PyStructSequence_SetItem(obj, 3, PyFloat_FromDouble(item.dx));
        PyStructSequence_SetItem(obj, 4, PyFloat_FromDouble(item.dy));
        PyStructSequence_SetItem(obj, 5, PyFloat_FromDouble(item.path));
        PyStructSequence_SetItem(obj, 6, indices);

        return obj;
    }

}  // namespace

PyObject *asora_test_create_lut([[maybe_unused]] PyObject *self, PyObject *args) {
    int q_max;
    int copy = true;
    if (!PyArg_ParseTuple(args, "i|p", &q_max, &copy)) return nullptr;

    std::vector<asora::lut_entry> lut;
    try {
        lut = asora::create_lut(q_max);
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }

    if (!copy) {
        // Return the size of the LUT if copy is false
        return PyLong_FromSize_t(lut.size());
    }

    PyObject *result = PyList_New(static_cast<Py_ssize_t>(lut.size()));
    if (!result) return nullptr;

    for (size_t i = 0; const auto &item : lut) {
        PyObject *entry = build_lut_entry(item);
        if (!entry) {
            Py_DECREF(result);
            return nullptr;
        }
        PyList_SET_ITEM(result, static_cast<Py_ssize_t>(i++), entry);
    }

    return result;
}

PyObject *asora_test_create_lut_edge_cases(
    [[maybe_unused]] PyObject *self, PyObject *args
) {
    if (!PyArg_ParseTuple(args, "")) return nullptr;

    std::vector<asora::lut_entry> lut;
    try {
        lut = asoratest::lut_edge_cases();
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }

    PyObject *result = PyList_New(static_cast<Py_ssize_t>(lut.size()));
    if (!result) return nullptr;

    for (size_t i = 0; const auto &item : lut) {
        PyObject *entry = build_lut_entry(item);
        if (!entry) {
            Py_DECREF(result);
            return nullptr;
        }
        PyList_SET_ITEM(result, static_cast<Py_ssize_t>(i++), entry);
    }

    return result;
}

PyObject *asora_test_cell_interpolator(
    [[maybe_unused]] PyObject *self, PyObject *args
) {
    PyArrayObject *dens;

    // Error checking
    if (!PyArg_ParseTuple(args, "O", &dens)) return nullptr;

    if (!PyArray_Check(dens) || PyArray_TYPE(dens) != NPY_DOUBLE ||
        PyArray_NDIM(dens) != 3) {
        PyErr_SetString(PyExc_TypeError, "dens must be numpy array of type double");
        return nullptr;
    }

    auto dens_data = static_cast<double *>(PyArray_DATA(dens));
    auto shape = PyArray_SHAPE(dens);

    auto coldens =
        reinterpret_cast<PyArrayObject *>(PyArray_SimpleNew(3, shape, NPY_DOUBLE));
    auto coldens_data = static_cast<double *>(PyArray_DATA(coldens));

    // Run test kernel
    try {
        std::array<size_t, 3> cpp_shape;
        std::copy(shape, shape + 3, cpp_shape.begin());
        asoratest::cell_interpolator(coldens_data, dens_data, cpp_shape);
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_MemoryError, e.what());
        return nullptr;
    }

    return PyArray_Return(reinterpret_cast<PyArrayObject *>(coldens));
}

PyObject *asora_test_geometric_factors(
    [[maybe_unused]] PyObject *self, PyObject *args
) {
    PyObject *shape_arg;
    std::array<size_t, 3> cpp_shape;

    // Error checking
    if (!PyArg_ParseTuple(args, "O", &shape_arg)) return nullptr;
    if (!PyArg_ParseTuple(
            shape_arg, "kkk", &cpp_shape[0], &cpp_shape[1], &cpp_shape[2]
        )) {
        PyErr_SetString(PyExc_TypeError, "only shape of dimension 3 is allowed");
        return nullptr;
    }

    std::array<npy_intp, 4> np_shape;
    std::copy(cpp_shape.begin(), cpp_shape.end(), np_shape.begin());
    np_shape[3] = 4;
    auto fact = reinterpret_cast<PyArrayObject *>(
        PyArray_SimpleNew(4, np_shape.data(), NPY_DOUBLE)
    );
    auto fact_data = static_cast<double *>(PyArray_DATA(fact));

    // Run test kernel
    try {
        asoratest::geometric_factors(fact_data, cpp_shape);
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_MemoryError, e.what());
        return nullptr;
    }

    return PyArray_Return(reinterpret_cast<PyArrayObject *>(fact));
}

PyObject *asora_test_path_in_cell([[maybe_unused]] PyObject *self, PyObject *args) {
    PyObject *shape_arg;
    std::array<size_t, 3> cpp_shape;

    // Error checking
    if (!PyArg_ParseTuple(args, "O", &shape_arg)) return nullptr;
    if (!PyArg_ParseTuple(
            shape_arg, "kkk", &cpp_shape[0], &cpp_shape[1], &cpp_shape[2]
        )) {
        PyErr_SetString(PyExc_TypeError, "only shape of dimension 3 is allowed");
        return nullptr;
    }

    std::array<npy_intp, 3> np_shape;
    std::copy(cpp_shape.begin(), cpp_shape.end(), np_shape.begin());
    auto path = reinterpret_cast<PyArrayObject *>(
        PyArray_SimpleNew(3, np_shape.data(), NPY_DOUBLE)
    );
    auto path_data = static_cast<double *>(PyArray_DATA(path));

    // Run test kernel
    try {
        asoratest::path_in_cell(path_data, cpp_shape);
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_MemoryError, e.what());
        return nullptr;
    }

    return PyArray_Return(reinterpret_cast<PyArrayObject *>(path));
}

PyObject *asora_test_linthrd2cart([[maybe_unused]] PyObject *self, PyObject *args) {
    int q, s;
    if (!PyArg_ParseTuple(args, "ii", &q, &s)) return nullptr;

    try {
        auto [i, j, k] = asoratest::linthrd2cart(q, s);
        return Py_BuildValue("iii", i, j, k);
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_MemoryError, e.what());
        return nullptr;
    }
}

PyObject *asora_test_cart2linthrd([[maybe_unused]] PyObject *self, PyObject *args) {
    int i, j, k;
    if (!PyArg_ParseTuple(args, "iii", &i, &j, &k)) return nullptr;

    try {
        auto [q, s] = asoratest::cart2linthrd(i, j, k);
        return Py_BuildValue("ii", q, s);
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_MemoryError, e.what());
        return nullptr;
    }
}

PyObject *asora_test_cells_in_shell([[maybe_unused]] PyObject *self, PyObject *args) {
    int q;
    if (!PyArg_ParseTuple(args, "i", &q)) return nullptr;

    auto n = asora::cells_in_shell(q);
    return Py_BuildValue("i", n);
}

PyObject *asora_test_cells_to_shell([[maybe_unused]] PyObject *self, PyObject *args) {
    int q;
    if (!PyArg_ParseTuple(args, "i", &q)) return nullptr;

    auto n = asora::cells_to_shell(q);
    return Py_BuildValue("i", n);
}

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// ========================================================================
// Define module functions and initialization function
// ========================================================================
static PyMethodDef asoraMethods[] = {
    {"cell_interpolator", asora_test_cell_interpolator, METH_VARARGS,
     "Test cell interpolation algorithm"},
    {"geometric_factors", asora_test_geometric_factors, METH_VARARGS,
     "Test geometric factors calculations"},
    {"path_in_cell", asora_test_path_in_cell, METH_VARARGS,
     "Test path-in-cell calculations"},
    {"linthrd2cart", asora_test_linthrd2cart, METH_VARARGS,
     "Shell indexing to cartesian coordinates"},
    {"cart2linthrd", asora_test_cart2linthrd, METH_VARARGS,
     "Cartesian coordinates to shell indexing"},
    {"cells_in_shell", asora_test_cells_in_shell, METH_VARARGS,
     "Number of cells in q-shell"},
    {"cells_to_shell", asora_test_cells_to_shell, METH_VARARGS,
     "Cumulative number of cells up to q-shell"},
    {"create_lut", asora_test_create_lut, METH_VARARGS,
     "Build the raytracing look-up table up to shell q_max"},
    {"create_lut_edge_cases", asora_test_create_lut_edge_cases, METH_VARARGS,
     "Build the edge case entires for teh raytracing look-up table"},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

static struct PyModuleDef asoramodule = {
    PyModuleDef_HEAD_INIT, "libasoratest",
    "Exposure of internal functions for testing purposes", -1, asoraMethods
};

PyMODINIT_FUNC PyInit_libasoratest(void) {
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
