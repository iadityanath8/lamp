// Copyright (C) 2024 Your Name <iadityanath8@gmail.com>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

#include "../include/ndarray.h"
#include <stdio.h>
#include <string.h>
#include <omp.h>
#include <immintrin.h>
#include <assert.h>
#include <stdbool.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <alloca.h>
#include <math.h>

static void compute_stride(ndarray *_arr)
{
    int n = _arr->dim;
    int *strd = _arr->strides;
    int *shape = _arr->shape;
    strd[n - 1] = 1;

    for (int i = n - 2; i >= 0; i--)
    {
        strd[i] = strd[i + 1] * shape[i + 1];
    }
}

ndarray *ndarray_init_with_arr(int *shape, int dim, float *arr)
{
    ndarray *_a = ALLOC(ndarray, 1);
    CHECK_ALLOC(_a);

    // _a->shape = ALLOC(int, dim);
    // _a->strides = ALLOC(int, dim);
    // CHECK_ALLOC(_a->shape);
    // CHECK_ALLOC(_a->strides);

    _a->dim = dim;
    size_t total_ele = 1;
    for (int i = 0; i < dim; i++)
    {
        _a->shape[i] = shape[i];
        total_ele *= shape[i];
    }

    _a->val = ALLOC(float, total_ele);
    CHECK_ALLOC(_a->val);
    _a->elements = total_ele;

    for (size_t i = 0; i < total_ele; i++)
    {
        _a->val[i] = arr[i];
    }

    compute_stride(_a);
    return _a;
}

NDARRAY_INLINE void ndarray_random(ndarray *_arr, float _min, float _max)
{
    if (!_arr || !_arr->val)
        return;

    float *a = _arr->val;

    for (size_t i = 0; i < _arr->elements; i++)
    {
        a[i] = _min + ((float)rand() / RAND_MAX) * (_max - _min);
    }
}

NDARRAY_INLINE ndarray *ndarray_zero(int *shape, int dim)
{
    ndarray *_c = ndarray_init(shape, dim);
    memset(_c->val, 0, sizeof(*_c->val) * _c->elements);
    return _c;
}

NDARRAY_INLINE ndarray *ndarray_ones(int *shape, int dim)
{
    ndarray *_c = ndarray_init(shape, dim);
    float* c = _c->val;
    for (size_t i = 0;i < _c->elements;i++) {
        *c = 1;c++;
    }
    return _c;
}

NDARRAY_INLINE ndarray *ndarray_random_(int *_shape, float *_v, int _dim)
{
    // int* shape = (int* )alloca(sizeof(int) * _dim);
    // memcpy(shape, _shape, sizeof(int)* _dim);

    ndarray *_crr = ndarray_init(_shape, _dim);
    // ndarray_random(_crr, _v[0],_v[1]);
    // printf("%f %f\n",_v[0],_v[1]);
    // printf("%d %d\n",_shape[0], _shape[1]);
    return _crr;
}

ndarray *ndarray_init_alloc_arr(int *shape, int dim, float *arr)
{
    ndarray *crr = ALLOC(ndarray, 1);
    CHECK_ALLOC(crr);

    crr->dim = dim;
    // crr->shape = ALLOC(int, dim);
    // crr->strides = ALLOC(int, dim);

    size_t total_ele = 1;
    for (int i = 0; i < dim; i++)
    {
        crr->shape[i] = shape[i];
        total_ele *= shape[i];
    }

    crr->elements = total_ele;
    crr->val = arr; // Caller must manage this memory
    compute_stride(crr);

    return crr;
}

ndarray ndarray_no_init(int *shape, int dim)
{
    ndarray _a; // = ALLOC(ndarray, 1);
    _a.dim = dim;

    size_t total_ele = 1;
    for (int i = 0; i < dim; i++)
    {
        _a.shape[i] = shape[i];
        total_ele *= _a.shape[i];
    }
    _a.elements = total_ele;
    _a.val = ALLOC(float, total_ele);

    compute_stride(&_a);
    return _a;
}

bool increment_indices(int* indices, int* shape, int dim) {
    for (int i = dim - 1; i >= 0; i--) {
        if (++indices[i] < shape[i]) return true;  
        indices[i] = 0;  
    }
    return false;  
}

ndarray* ndarray_slice(ndarray* _arr, int* start, int* end, int dim) {
    ASSERT(dim == _arr->dim);
    
    ndarray* _brr = ALLOC(ndarray, 1);
    _brr->dim = dim;

    int offset = 0;
    int total_elements = 1;
    
    for (int i = 0; i < dim; i++) {
        offset += start[i] * _arr->strides[i];  
        _brr->shape[i] = end[i] - start[i];    
        _brr->strides[i] = _arr->strides[i];  
        total_elements *= _brr->shape[i];
    }
    
    _brr->elements = total_elements;
    
    _brr->val = ALLOC(float, total_elements);
    
    int indices[MAX_DIM] = {0};  
    int i = 0;
    do {
        int src_offset = offset;
        for (int j = 0; j < dim; j++) {
            src_offset += indices[j] * _arr->strides[j];
        }
        _brr->val[i++] = _arr->val[src_offset];
    } while (increment_indices(indices, _brr->shape, dim));
    
    return _brr;
}

ndarray* ndarray_get(ndarray* _arr, int* shape,int dim) {
    _inner_items arr = _arr->val;
    int _dim         = _arr->dim;
    ASSERT(dim <= _dim);
    int* strd        = _arr->strides;
    int* shp         = _arr->shape;
    int  n           = _arr->elements;
    int nd           = _arr->dim - dim;
    int total        = 1;

    ndarray* _brr    = ALLOC(ndarray,1);
    int* bshape      = _brr->shape;
    int* bstrd       = _brr->strides;
    _brr->dim        = nd;

    int offset = 0;
    for (int i = 0;i < dim;i++) offset += shape[i] * strd[i];
    
    for (int i = dim;i < _dim;i++) {
      bshape[i-dim] = shp[i];
      bstrd[i-dim]  = strd[i];
      total *= bshape[i-dim];
    }
    _brr->val = _arr->val + offset;
    _brr->elements = total;
    return _brr;
}

/** Powered by gpu using cublas library */

int matmul__cublas_(float *h_A, float *h_B, float *h_C, 
                     int M, int N, int K, 
                     bool trans_A, bool trans_B)
{
    cublasHandle_t handle;
    cublasStatus_t status;

    status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    }

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size_A);
    cudaMalloc((void **)&d_B, size_B);
    cudaMalloc((void **)&d_C, size_C);

    cublasSetMatrix(M, K, sizeof(float), h_A, M, d_A, M);
    cublasSetMatrix(K, N, sizeof(float), h_B, K, d_B, K);

    float alpha = 1.0f;
    float beta = 0.0f;

    // Set cuBLAS operation flags based on transposition
    cublasOperation_t op_A = trans_A ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t op_B = trans_B ? CUBLAS_OP_T : CUBLAS_OP_N;

    // Perform matrix multiplication: C = A * B
    cublasSgemm(handle, op_B, op_A,  // Note: cuBLAS uses column-major order
                N, M, K,
                &alpha, d_B, trans_B ? K : N,
                        d_A, trans_A ? M : K,
                &beta,  d_C, N);

    cublasGetMatrix(M, N, sizeof(float), d_C, M, h_C, M);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);

    return 0;
}


ndarray *ndarray_init(int *shape, int dim)
{
    ndarray *_a = ALLOC(ndarray, 1);
    _a->dim = dim;

    size_t total_ele = 1;
    for (int i = 0; i < dim; i++)
    {
        _a->shape[i] = shape[i];
        total_ele *= _a->shape[i];
    }
    _a->elements = total_ele;
    _a->val = ALLOC(float, total_ele);

    compute_stride(_a);
    return _a;
}

NDARRAY_INLINE void ndarray_fill(ndarray *_arr, float _v)
{
    int n = _arr->elements;
    float *a = _arr->val;

    for (size_t i = 0; i < n; i++)
    {
        *a = _v;
        a++;
    }
}

NDARRAY_INLINE float ndarray_get_element(ndarray *arr, int *shape)
{
    size_t offset = 0;
    int *strd = arr->strides;

    if (arr->dim <= 4)
    {
        switch (arr->dim)
        {
        case 4:
            offset += strd[3] * shape[3];
        case 3:
            offset += strd[2] * shape[2];
        case 2:
            offset += strd[1] * shape[1];
        case 1:
            offset += strd[0] * shape[0];
        }
    }
    else
    {
        for (size_t i = 0; i < arr->dim; i++)
        {
            offset += strd[i] * shape[i];
        }
    }

    ASSERT(offset < arr->elements);
    return arr->val[offset];
}

NDARRAY_INLINE void ndarray_put(ndarray *arr, int *shape, float _val)
{
    size_t offset = 0;
    int *strd = arr->strides;
    if (arr->dim <= 4)
    {
        switch (arr->dim)
        {
        case 4:
            offset += strd[3] * shape[3];
        case 3:
            offset += strd[2] * shape[2];
        case 2:
            offset += strd[1] * shape[1];
        case 1:
            offset += strd[0] * shape[0];
        default:
            break;
        }
    }
    else
    {
        for (size_t i = 0; i < arr->dim; i++)
        {
            offset += strd[i] * shape[i];
        }
    }

    assert(offset < arr->elements);
    arr->val[offset] = _val;
}

NDARRAY_INLINE void ndarray_add_scalar(ndarray *_arr, float _adder)
{
    float *v = _arr->val;
    for (size_t i = 0; i < _arr->elements; i++)
    {
        v[i] += _adder;
    }
}

NDARRAY_INLINE ndarray *ndarray_add_vector(ndarray *_arr, ndarray *_brr)
{
    ASSERT_SAME_SHAPE(_arr, _brr);

    ndarray *_crr = ndarray_init(_arr->shape, _arr->dim);

    float *a = _arr->val;
    float *b = _brr->val;
    float *c = _crr->val;

    for (size_t i = 0; i < _crr->elements; i++)
    {
        c[i] = a[i] + b[i];
    }
    return _crr;
}

NDARRAY_INLINE void ndarray_add_vector_inplace(ndarray *_arr, ndarray *_brr)
{
    ASSERT_SAME_SHAPE(_arr, _brr);
    float *a = _arr->val;
    float *b = _brr->val;

    for (size_t i = 0; i < _arr->elements; i++)
    {
        a[i] += b[i]; // remove pointer indirection by using stack allocation and do not incur stupid mulitpication and addition
    }
}

NDARRAY_INLINE void ndarray_mul_scalar(ndarray *_arr, float _mul)
{
    float *a = _arr->val;
    for (size_t i = 0; i < _arr->elements; i++)
    {
        a[i] *= _mul;
    }
}

ndarray *ndarray_mul_vector(ndarray *_arr, ndarray *_brr)
{
    ASSERT_SAME_SHAPE(_arr, _brr);
    ndarray *_crr = ndarray_init(_arr->shape, _arr->dim);

    float *a = _arr->val;
    float *b = _brr->val;
    float *c = _crr->val;

    for (size_t i = 0; i < _crr->elements; i++)
    {
        c[i] = a[i] * b[i];
    }
    return _crr;
}

NDARRAY_INLINE void ndarray_mul_vector_inplace(ndarray *_arr, ndarray *_brr)
{
    ASSERT_SAME_SHAPE(_arr, _brr);
    float *a = _arr->val;
    float *b = _brr->val;

    for (size_t i = 0; i < _arr->elements; i++)
    {
        a[i] *= b[i];
    }
}

NDARRAY_INLINE void ndarray_div_scalar(ndarray *_arr, float _div)
{
    assert(_div != 0);

    float *a = _arr->val;
    for (size_t i = 0; i < _arr->elements; i++)
    {
        a[i] /= _div;
    }
}

ndarray *ndarray_div_vector(ndarray *_arr, ndarray *_brr)
{
    ASSERT_SAME_SHAPE(_arr, _brr);

    ndarray *_crr = ndarray_init(_arr->shape, _arr->dim);

    float *c = _crr->val;
    float *a = _arr->val;
    float *b = _brr->val;

    for (size_t i = 0; i < _crr->elements; i++)
    {
        c[i] = a[i] / b[i];
    }
    return _crr;
}

NDARRAY_INLINE void ndarray_div_vector_inplace(ndarray *_arr, ndarray *_brr)
{
    ASSERT_SAME_SHAPE(_arr, _brr);

    float *a = _arr->val;
    float *b = _brr->val;

    for (size_t i = 0; i < _arr->elements; i++)
    {
        a[i] /= b[i];
    }
}

NDARRAY_INLINE ndarray *ndarray_pow(ndarray *_arr, float _pow)
{
    ndarray *_crr = ndarray_init(_arr->shape, _arr->dim);
    float *a = _arr->val;
    float *c = _crr->val;
    int n = _arr->elements;

    for (size_t i = 0; i < n; i++)
    {
        c[i] = powf(a[i], _pow);
    }
    return _crr;
}

NDARRAY_INLINE void ndarray_pow_inplace(ndarray *_arr, float _pow)
{
    int n = _arr->elements;
    float *a = _arr->val;

    for (size_t i = 0; i < n; i++)
    {
        a[i] = powf(a[i], _pow);
    }
}

NDARRAY_INLINE ndarray *ndarray_negate(ndarray *_arr)
{
    ndarray *_crr = ndarray_init(_arr->shape, _arr->dim);

    int n = _arr->elements;
    float *a = _arr->val;
    float *c = _crr->val;

    for (size_t i = 0; i < n; i++)
    {
        c[i] = -a[i];
    }
    return _crr;
}

NDARRAY_INLINE void ndarray_negate_inplace(ndarray *_arr)
{
    float *a = _arr->val;
    int n = _arr->elements;

    for (size_t i = 0; i < n; i++)
    {
        *a = -*a;
        a++;
    }
}

void naive_matmul(float *A, float *B, float *C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0;
            for (int p = 0; p < K; p++) {
                sum += A[i * K + p] * B[p * N + j];  
            }
            C[i * N + j] = sum;
        }
    }
}

// Function to perform matrix multiplication
/** Now it is using cublas as its backend to compute matrix multiplication ssoon it will role its custom
 *  matrix multiplication using its own version as well as it will soon procvide openblas or intel MKL
 *  version
 */
ndarray *ndarray_matmul(ndarray *_arr, ndarray *_brr,bool ta,bool tb)
{
    assert(_arr->shape[1] == _brr->shape[0]);

    int N = _arr->shape[0];
    int M = _brr->shape[1];
    int K = _arr->shape[1];

    ndarray *_crr = ndarray_init((int[]){N, M}, 2);

    // Assuming row-major order, set transA and transB to false
    matmul__cublas_(_arr->val, _brr->val, _crr->val, N, M, K, ta, tb);
    return _crr;
}


void ndarray_dump(const struct ndarray *arr)
{
    printf("ndarray Dump:\n");

    printf("  Shape: (");
    for (int i = 0; i < arr->dim; i++)
    {
        printf("%d", arr->shape[i]);
        if (i < arr->dim - 1)
            printf(", ");
    }
    printf(")\n");
    printf("  Strides: (");
    for (int i = 0; i < arr->dim; i++)
    {
        printf("%d", arr->strides[i]);
        if (i < arr->dim - 1)
            printf(", ");
    }
    printf(")\n");

    printf("  Values:\n");

    float *data = arr->val;
    int total_elements = arr->elements;

    for (int i = 0; i < total_elements; i++)
    {
        printf("    [%d] = %f\n", i, data[i]);
    }
}

__attribute__((deprecated))
void ndarray_transpose(ndarray *arr)
{
    if (arr->dim < 2)
        return; // No need to transpose a 1D array

    for (int i = 0; i < arr->dim / 2; i++)
    {
        int j = arr->dim - i - 1;

        int temp = arr->shape[i];
        arr->shape[i] = arr->shape[j];
        arr->shape[j] = temp;

        temp = arr->strides[i];
        arr->strides[i] = arr->strides[j];
        arr->strides[j] = temp;
    }
}

NDARRAY_INLINE void ndarray_matrix_transpose_inplace(ndarray* _arr) {
    ASSERT(_arr->dim == 2);

    int temp = _arr->shape[0];
    _arr->shape[0] = _arr->shape[1];
    _arr->shape[1] = temp;

    temp = _arr->strides[0];
    _arr->strides[0] = _arr->strides[1];
    _arr->strides[1] = temp;
}


ndarray *ndarray_clone(ndarray *_arr) {
    if (_arr->elements <= 0) {
        fprintf(stderr, "ndarray_clone: Invalid element count\n");
        return NULL;
    }

    ndarray *_crr = ndarray_init(_arr->shape, _arr->dim);
    _crr->elements = _arr->elements;
    memcpy(_crr->val, _arr->val, sizeof(float) * _arr->elements);    
    memcpy(_crr->strides, _arr->strides, sizeof(int) * _arr->dim);
    return _crr;
}


ndarray* ndarray_matrix_transpose(ndarray* _arr) {
    ndarray* _cl = ndarray_clone(_arr);
    ndarray_matrix_transpose_inplace(_cl);
    return _cl;
}

void ndarray_free(ndarray *_arr)
{
    DEALLOC(_arr->val);
    DEALLOC(_arr);
}
