// Copyright (C) 2024 Nemesis <iadityanath8@gmail.com>
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


#ifndef NDARRAY_H
#define NDARRAY_H

#include <stdlib.h>
#include <assert.h>
#include "utils.h"
#include <stdio.h>
#include <stdbool.h>
#define NDARRAY_INLINE inline __attribute__((always_inline))
#define MAX_DIM 10

typedef struct ndarray ndarray;
typedef float *_inner_items;
 
// basic ndarray struct in here
struct ndarray
{
    /** inner_items used for storing the values in float32 format
     *  as it is 32 bit and simd avx can store 8 of these in a avx2 and 16
     *  of these in avx512 based system so float is preferred in this for high
     *  precision double of f64 is recommended
     */
    _inner_items val;

    /** MAX support dimension is 10 */
    int shape[MAX_DIM];
    int strides[MAX_DIM];
    
    /**  (x,y,z,....) dimensions **/
    int dim;
    int elements;
};

/** computing the stride in here ok  */

ndarray *ndarray_init_with_arr(int *shape, int dim, float *arr);
ndarray *ndarray_init(int *shape, int dim);
ndarray *ndarray_init_alloc_arr(int *shape, int dim, float *arr); // allocated arr
ndarray ndarray_no_init(int *shape, int dim);
ndarray* ndarray_clone(ndarray* _arr);

/** allocates memory on the heap user have to free for this **/
ndarray* ndarray_slice(ndarray* _arr, int* start, int* end, int dim);


ndarray* ndarray_zero(int* shape, int dim);
ndarray* ndarray_ones(int* shape, int dim);
//__attribute__((deprecated)) void ndarray_transpose(ndarray* _arr);
/** Assumming this is a matrix  **/
ndarray* ndarray_matrix_transpose(ndarray* _arr);
void ndarray_matrix_transpose_inplace(ndarray* _arr);

void ndarray_fill(ndarray *_arr, float _v);

float ndarray_get_element(ndarray *arr, int *shape);
void ndarray_put(ndarray *arr, int *shape, float _val);
ndarray* ndarray_get(ndarray* _arr, int* shape,int dim);

void ndarray_add_scalar(ndarray *_arr, float _adder);
ndarray *ndarray_add_vector(ndarray *_arr, ndarray *_brr);
void ndarray_add_vector_inplace(ndarray *_arr, ndarray *_brr);

void ndarray_mul_scalar(ndarray *_arr, float _mul);
ndarray *ndarray_mul_vector(ndarray *_arr, ndarray *_brr);
void ndarray_mul_vector_inplace(ndarray *_arr, ndarray *_brr);

void ndarray_div_scalar(ndarray *_arr, float _div);
ndarray *ndarray_div_vector(ndarray *_arr, ndarray *_brr);
void ndarray_div_vector_inplace(ndarray *_arr, ndarray *_brr);

/** unary operations */
ndarray *ndarray_negate(ndarray *_arr);
void ndarray_negate_inplace(ndarray *_arr);

ndarray *ndarray_pow(ndarray *_arr, float pow);
void ndarray_pow_inplace(ndarray *_arr, float pow);

/** Only 2d matrix operations */
ndarray *ndarray_matmul(ndarray *_arr, ndarray *_brr,bool ta, bool tb);
__attribute__((deprecated)) void matmul_tiled(float *A, float *B, float *C, int rowsA, int colsB, int colsA);
//__attribute__((deprecated)) void matmul_fast(float *restrict A, float *restrict B, float *restrict C, size_t rowsA, size_t colsA, size_t colsB);

void ndarray_free(ndarray *_arr);

/** only for debugging purpose in here **/
void ndarray_dump(const struct ndarray *arr);

void ndarray_random(ndarray* _arr, float _min, float _max);
ndarray* ndarray_random_(int* _shape, float* _vals,int _dim);
#endif
