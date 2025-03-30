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

#include "../include/tensor.h"
#include <stdbool.h>
tensor *tensor_init(ndarray *_arr)
{
    tensor *a = ALLOC(tensor, 1);
    a->data = _arr;
    a->grad = NULL;
    a->childrens = NULL;
    a->user_data = NULL;
    a->num_childrens = 0;
    return a;
}

tensor *tensor_init_with_arr(float *_arr, int *shape, int dim)
{
    ndarray *_crr = ndarray_init_with_arr(SHP(shape[0], shape[1]), dim, _arr);
    tensor *t = tensor_init(_crr);
    return t;
}

void dump_tensor(const tensor *t, int indent_level)
{
    // Create indentation strings
    char indent[32] = {0};
    char child_indent[32] = {0};

    for (int i = 0; i < indent_level && i < 31; i++)
    {
        indent[i] = ' ';
    }
    for (int i = 0; i < indent_level + 2 && i < 31; i++)
    {
        child_indent[i] = ' ';
    }

    printf("%sTensor @ %p {\n", indent, (void *)t);

    printf("%s  data: ", indent);
    if (t->data)
    {
        printf("\n%s", child_indent);
        ndarray_dump(t->data);
    }
    else
    {
        printf("NULL\n");
    }

    printf("%s  grad: ", indent);
    if (t->grad)
    {
        printf("\n%s", child_indent);
        ndarray_dump(t->grad);
    }
    else
    {
        printf("NULL\n");
    }

    // Print backward function info
    printf("%s  backwardFn: %p\n", indent, (void *)t->bk);

    // Print children
    printf("%s  children (%zu): [\n", indent, t->num_childrens);
    for (size_t i = 0; i < t->num_childrens; i++)
    {
        if (t->childrens[i])
        {
            dump_tensor(t->childrens[i], indent_level + 4);
        }
        else
        {
            printf("%s    NULL\n", indent);
        }
        if (i < t->num_childrens - 1)
        {
            printf("%s    ---\n", indent);
        }
    }
    printf("%s  ]\n", indent);

    // Print user data
    printf("%s  user_data: %p\n", indent, t->user_data);

    printf("%s}\n", indent);
}

/** * TODO: thinking of inlining the functions  */
void add_backward(tensor *self)
{
    tensor **chld = self->childrens;
    ndarray *nd = self->data;
    for (size_t i = 0; i < self->num_childrens; i++)
    {
        tensor *child = chld[i];
        if (child->grad == NULL)
        {
            child->grad = ndarray_zero(nd->shape, nd->dim);
        }
        ndarray_add_vector_inplace(child->grad, self->grad);
    }
}

void mul_backward(tensor *self)
{
    tensor **chld = self->childrens;
    ndarray *z = self->data; 
    ndarray *grad_z = self->grad;

    for (size_t i = 0; i < self->num_childrens; i++)
    {
        tensor *child = chld[i];
        ndarray *x_i = child->data;

        if (child->grad == NULL)
        {
            child->grad = ndarray_zero(z->shape, z->dim);
        }

        ndarray *temp = ndarray_div_vector(z, x_i); 
        ndarray_mul_vector_inplace(temp, grad_z); 
        ndarray_add_vector_inplace(child->grad, temp);

        ndarray_free(temp);
    }
}

void matmul_backward(tensor* self) {
    tensor* a = self->childrens[0];
    tensor* b = self->childrens[1];
    ndarray* grad_out = self->grad;


    if (!a->grad) {
        a->grad = ndarray_zero(a->data->shape, a->data->dim);
    }
    if (!b->grad) {
        b->grad = ndarray_zero(b->data->shape, b->data->dim);  
    }

    // dA = grad_out @ B^T
    ndarray* b_trans = ndarray_matrix_transpose(b->data);
    ndarray* grad_a = ndarray_matmul(grad_out, b_trans,false, true);
    ndarray_add_vector_inplace(a->grad, grad_a);
    ndarray_free(b_trans);
    ndarray_free(grad_a);
        
    // dB = A^T @ grad_out
    ndarray* a_trans = ndarray_matrix_transpose(a->data);
    ndarray* grad_b = ndarray_matmul(a_trans, grad_out,true,false);
    ndarray_add_vector_inplace(b->grad, grad_b);
    ndarray_free(a_trans);
    ndarray_free(grad_b);
}

tensor *t_add(int n, ...)
{
    va_list args;
    va_start(args, n);

    tensor *t = va_arg(args, tensor *);
    ndarray *v_ = t->data;

    ndarray *sum = ndarray_zero(v_->shape, v_->dim);

    ndarray_add_vector_inplace(sum, v_);
    _Many(tensor *) childrens = ALLOC(tensor *, n);
    childrens[0] = t;

    for (int i = 1; i < n; i++)
    {
        tensor *_t = va_arg(args, tensor *);
        ndarray *_v = _t->data;

        ndarray_add_vector_inplace(sum, _v);
        childrens[i] = _t;
    }

    tensor *res = tensor_init(sum);
    res->childrens = childrens;
    res->num_childrens = n;
    res->bk = add_backward;
    return res;
}

tensor *t_mul(int n, ...)
{
    va_list args;
    va_start(args, n);

    tensor **childrens = ALLOC(tensor *, n);

    // ndarray* mul
    tensor *t = va_arg(args, tensor *);
    ndarray *v = t->data;
    ndarray *_m = ndarray_ones(v->shape, v->dim);
    ndarray_mul_vector_inplace(_m, v);

    childrens[0] = t;

    for (size_t i = 1; i < n; i++)
    {
        tensor *t = va_arg(args, tensor *);
        ndarray *v = t->data;
        ndarray_mul_vector_inplace(_m, v);
        childrens[i] = t;
    }
    tensor *res = tensor_init(_m);
    res->childrens = childrens;
    res->num_childrens = n;
    res->bk = mul_backward;
    return res;
}

tensor* t_matmul(tensor* a, tensor* b) {
    ASSERT(a->data->dim == 2 && b->data->dim == 2);
    ndarray*c =  ndarray_matmul(a->data,b->data,false,false);        
    tensor* out = tensor_init(c);

    out->childrens = ALLOC(tensor* , 2);
    out->num_childrens = 2;
    out->childrens[0] = a;
    out->childrens[1] = b;
    out->bk = matmul_backward;
    return out;
}   
