#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_SSIZE_T_CLEAN

#include "memory.cuh"
#include "raytracing.cuh"

#include <Python.h>
#include <numpy/arrayobject.h>

// ===========================================================================
// ASORA Python C-extension module
// Mostly boilerplate code, this file contains the wrappers for python
// to access the C++ functions of the ASORA library. Care has to be taken
// mostly with the numpy array arguments, since the underlying raw C pointer
// is passed directly to the C++ functions without additional type checking.
// ===========================================================================

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
    PyObject *numpy_array_to_device(const PyArrayObject *array, T *&dst_dev) {
        if (!PyArray_Check(array) || PyArray_TYPE(array) != getNpyType<T>()) {
            using namespace std::string_literals;
            std::string msg =
                "array must be a numpy NDArray of type "s + typeid(T).name();
            PyErr_SetString(PyExc_TypeError, msg.c_str());
            return NULL;
        }

        auto data = static_cast<T *>(PyArray_DATA(array));
        auto size = static_cast<size_t>(PyArray_NBYTES(array));
        asora::array_to_device(dst_dev, data, size);
        return Py_None;
    }

}  // namespace

// ========================================================================
// Raytrace all sources and compute photoionization rates
// ========================================================================
static PyObject *asora_do_all_sources(PyObject *self, PyObject *args) {
    double R;
    double sig;
    double dr;
    PyArrayObject *xh_av;
    PyArrayObject *phi_ion;
    int NumSrc;
    int m1;
    double minlogtau;
    double dlogtau;
    int num_tau;

    if (!PyArg_ParseTuple(
            args, "dddOOiiddi", &R, &sig, &dr, &xh_av, &phi_ion, &NumSrc, &m1,
            &minlogtau, &dlogtau, &num_tau
        ))
        return NULL;

    // Error checking
    if (!PyArray_Check(xh_av) || PyArray_TYPE(xh_av) != NPY_DOUBLE) {
        PyErr_SetString(PyExc_TypeError, "xh_av must be Array of type double");
        return NULL;
    }
    if (!PyArray_Check(phi_ion) || PyArray_TYPE(phi_ion) != NPY_DOUBLE) {
        PyErr_SetString(PyExc_TypeError, "phi_ion must be Array of type double");
        return NULL;
    }

    // Get Array data
    auto xh_av_data = static_cast<double *>(PyArray_DATA(xh_av));
    auto phi_ion_data = static_cast<double *>(PyArray_DATA(phi_ion));

    asora::do_all_sources_gpu(
        R, sig, dr, xh_av_data, phi_ion_data, NumSrc, m1, minlogtau, dlogtau, num_tau
    );

    return Py_None;
}

// ========================================================================
// Allocate GPU memory for grid data
// ========================================================================
static PyObject *asora_device_init(PyObject *self, PyObject *args) {
    int N;
    int num_src_par;
    int mpi_rank;
    int num_gpus;
    if (!PyArg_ParseTuple(args, "iiii", &N, &num_src_par, &mpi_rank, &num_gpus))
        return NULL;

    asora::device_init(N, num_src_par, mpi_rank, num_gpus);
    return Py_None;
}

// ========================================================================
// Deallocate GPU memory
// ========================================================================
static PyObject *asora_device_close(PyObject *self, PyObject *args) {
    asora::device_close();
    return Py_None;
}

// ========================================================================
// Copy density grid to GPU
// ========================================================================
static PyObject *asora_density_to_device(PyObject *self, PyObject *args) {
    PyArrayObject *ndens;
    if (!PyArg_ParseTuple(args, "O", &ndens)) return NULL;

    if (auto ret = numpy_array_to_device(ndens, asora::n_dev); !ret) return nullptr;

    return Py_None;
}

// ========================================================================
// Copy radiation table to GPU
// ========================================================================
static PyObject *asora_photo_table_to_device(PyObject *self, PyObject *args) {
    PyArrayObject *thin_table;
    PyArrayObject *thick_table;
    if (!PyArg_ParseTuple(args, "OO", &thin_table, &thick_table)) return NULL;

    if (auto ret = numpy_array_to_device(thin_table, asora::photo_thin_table_dev); !ret)
        return nullptr;
    if (auto ret = numpy_array_to_device(thick_table, asora::photo_thick_table_dev);
        !ret)
        return nullptr;

    return Py_None;
}

// ========================================================================
// Copy source data to GPU
// ========================================================================
static PyObject *asora_source_data_to_device(PyObject *self, PyObject *args) {
    PyArrayObject *pos;
    PyArrayObject *flux;
    if (!PyArg_ParseTuple(args, "OO", &pos, &flux)) return NULL;

    if (auto ret = numpy_array_to_device(pos, asora::src_pos_dev); !ret) return nullptr;
    if (auto ret = numpy_array_to_device(flux, asora::src_flux_dev); !ret)
        return nullptr;

    return Py_None;
}

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// ========================================================================
// Define module functions and initialization function
// ========================================================================
static PyMethodDef asoraMethods[] = {
    {"do_all_sources", asora_do_all_sources, METH_VARARGS, "Do OCTA raytracing (GPU)"},
    {"device_init", asora_device_init, METH_VARARGS, "Free GPU memory"},
    {"device_close", asora_device_close, METH_VARARGS, "Free GPU memory"},
    {"density_to_device", asora_density_to_device, METH_VARARGS,
     "Copy density field to GPU"},
    {"photo_table_to_device", asora_photo_table_to_device, METH_VARARGS,
     "Copy radiation table to GPU"},
    {"source_data_to_device", asora_source_data_to_device, METH_VARARGS,
     "Copy radiation table to GPU"},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

static struct PyModuleDef asoramodule = {
    PyModuleDef_HEAD_INIT, "libasora",                         /* name of module */
    "CUDA C++ implementation of the short-characteristics RT", /* module
                                                                  documentation,
                                                                  may be NULL */
    -1, /* size of per-interpreter state of the module,
          or -1 if the module keeps state in global variables. */
    asoraMethods
};

PyMODINIT_FUNC PyInit_libasora(void) {
    PyObject *module = PyModule_Create(&asoramodule);
    import_array();
    return module;
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
