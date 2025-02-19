#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdint.h>
#include <iostream>
#include <complex>
#include "integration.hpp"

static PyObject *integral(PyObject *self, PyObject *args);

/////// Python-module-related functions and tables

// The module's method table
static PyMethodDef _integrationMethods[] = {
    {"integral", integral, METH_VARARGS, "Compute the integral using C++."},
    {NULL, NULL, 0, NULL}
};

// The module definition function
static struct PyModuleDef _integration = {
    PyModuleDef_HEAD_INIT,
    "_integration",
    NULL, // Module documentation
    -1,
    _integrationMethods
};

// The module initialization function
PyMODINIT_FUNC PyInit__integration(void) { 
    import_array(); // Ensure NumPy is properly initialized
    return PyModule_Create(&_integration);
}

//////// The actual functions of the modules

static PyObject *integral(PyObject *self, PyObject *args) {
    int32_t M, k;
    double delta;
    PyObject *A_o;

    // Parse Python arguments
    if (!PyArg_ParseTuple(args, "Odi", &A_o, &delta, &k)) {
        return NULL;
    }

    // Convert input to NumPy array
    PyArrayObject *A_a = (PyArrayObject *) PyArray_FromAny(
        A_o, NULL, 0, 0, NPY_ARRAY_CARRAY, NULL
    );

    if (A_a == NULL) {
        PyErr_SetString(PyExc_TypeError, "Failed to convert input to NumPy array.");
        return NULL;
    }

    // Get array descriptor (data type information)
    PyArray_Descr *info = PyArray_DESCR(A_a);
    if (!info) {
        Py_DECREF(A_a);
        PyErr_SetString(PyExc_TypeError, "Could not retrieve NumPy array descriptor.");
        return NULL;
    }

    // Get the length of the array (first dimension)
    M = (int32_t) PyArray_DIM(A_a, 0);

    // Result object
    PyObject *result = NULL;

    // Determine type and call corresponding template function
    if (PyDataType_ISFLOAT(info)) {
        double *A = (double *) PyArray_DATA(A_a);
        double res = integration::integral<double>(A, M, delta, k);
        result = Py_BuildValue("d", res);
    } 
    else if (PyDataType_ISCOMPLEX(info)) {
        std::complex<double> *A = (std::complex<double> *) PyArray_DATA(A_a);
        std::complex<double> res = integration::integral<std::complex<double>>(A, M, delta, k);
        result = PyComplex_FromDoubles(res.real(), res.imag());
    } 
    else {
        PyErr_SetString(PyExc_TypeError, "Unsupported data type.");
        result = NULL;
    }

    // Cleanup
    Py_DECREF(A_a);

    return result;
}
