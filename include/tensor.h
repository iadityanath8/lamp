// Copyright (C) 2024 Aditya <iadityanath8@gmail.com>
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

#ifndef TENSOR_H
#define TENSOR_H

#include "ndarray.h"
#include "utils.h"
#include <stdarg.h>

#define COUNT_ARGS(...) (sizeof((tensor *[]){__VA_ARGS__}) / sizeof(tensor *))
#define T_ADD(...) t_add(COUNT_ARGS(__VA_ARGS__),__VA_ARGS__)
#define T_MUL(...) t_mul(COUNT_ARGS(__VA_ARGS__),__VA_ARGS__)

typedef struct tensor tensor;
typedef void (*backwarfFn)(tensor *);

struct tensor {
    /** 
    *  tensor object holds the value for the generic 
    *  ndarray structure as a gradient and the main data 
    *  provides effcient tape based backwardFn and 
    *  Fully easy to use ABI  
    */

    ndarray* data;

    /** gradient ndarray will store the gradient in the ndarray 
     *  form which will be used for making computation easy and fast */
    /** gradient ndarray will store the gradient in the ndarray 
     *  form which will be used for making computation easy and fast */
    ndarray* grad;
    backwarfFn bk;
    _Many(tensor*) childrens;
    size_t num_childrens;
    void* user_data;
};

void add_backward(tensor* self);

tensor* tensor_init(ndarray* nd);
tensor* tensor_init_with_arr(float* _arr, int* shape, int dim);
tensor* t_add(int n, ...);
tensor* t_sub(int n, ...);
tensor* t_mul(int n, ...);
tensor* t_div(int n, ...);

/* right now only 2 parameter with only matrix support*/
tensor* t_matmul(tensor* a, tensor* b);

void dump_tensor(const tensor* t, int indent_level);


#endif 