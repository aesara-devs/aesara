#section support_code_apply

int APPLY_SPECIFIC(cpu_dimshuffle)(PyArrayObject *input, PyArrayObject **res,
                                   PARAMS_TYPE *params) {

  // This points to either the original input or a copy we create below.
  // Either way, this is what we should be working on/with.
  PyArrayObject *_input;

  if (*res)
    Py_XDECREF(*res);

  if (params->inplace) {
    _input = input;
    Py_INCREF((PyObject *)_input);
  } else {
    _input = (PyArrayObject *)PyArray_FromAny(
        (PyObject *)input, NULL, 0, 0, NPY_ARRAY_ALIGNED | NPY_ARRAY_ENSURECOPY,
        NULL);
  }

  PyArray_Dims permute;

  if (!PyArray_IntpConverter((PyObject *)params->transposition, &permute)) {
    return 1;
  }

  /*
    res = res.transpose(self.transposition)
  */
  PyArrayObject *transposed_input =
      (PyArrayObject *)PyArray_Transpose(_input, &permute);

  Py_DECREF(_input);

  PyDimMem_FREE(permute.ptr);

  npy_intp *res_shape = PyArray_DIMS(transposed_input);
  npy_intp N_shuffle = PyArray_SIZE(params->shuffle);
  npy_intp N_augment = PyArray_SIZE(params->augment);
  npy_intp N = N_augment + N_shuffle;
  npy_intp *_reshape_shape = PyDimMem_NEW(N);

  if (_reshape_shape == NULL) {
    PyErr_NoMemory();
    return 1;
  }

  /*
    shape = list(res.shape[: len(self.shuffle)])
    for augm in self.augment:
        shape.insert(augm, 1)
  */
  npy_intp aug_idx = 0;
  int res_idx = 0;
  for (npy_intp i = 0; i < N; i++) {
    if (aug_idx < N_augment &&
        i == *((npy_intp *)PyArray_GetPtr(params->augment, &aug_idx))) {
      _reshape_shape[i] = 1;
      aug_idx++;
    } else {
      _reshape_shape[i] = res_shape[res_idx];
      res_idx++;
    }
  }

  PyArray_Dims reshape_shape = {.ptr = _reshape_shape, .len = (int)N};

  /* res = res.reshape(shape) */
  *res = (PyArrayObject *)PyArray_Newshape(transposed_input, &reshape_shape,
                                           NPY_CORDER);

  Py_DECREF(transposed_input);

  PyDimMem_FREE(reshape_shape.ptr);

  if (!*res) {
    return 1;
  }

  return 0;
}
