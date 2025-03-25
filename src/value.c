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


#include "../include/value.h"
#include <stdlib.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

/** TODO: soon we will try to make it explicit we made it like this because of the issue of the verbosity */
tape GLOBAL_TAPE = {.items = NULL, .count = 0, .capacity = 0, .require_grad = false};

inline void tape_init()
{
    GLOBAL_TAPE.require_grad = true;
}

inline void tape_deinit()
{
    GLOBAL_TAPE.require_grad = false;
}

void dump_tape()
{
    for (int i = 0; i < GLOBAL_TAPE.count; i++)
    {
        printf("%f ", GLOBAL_TAPE.items[i]->data);
        printf("Num of childrens -> \n");

        for (int j = 0; j < GLOBAL_TAPE.items[i]->num_childrens; j++)
        {
            printf("%f\n", GLOBAL_TAPE.items[i]->childrens[j]->data);
        }
    }
}

value *value_init(float _d)
{
    value *_v = ALLOC(value, 1);
    _v->bk = NULL;
    _v->childrens = NULL;
    _v->data = _d;
    _v->_grad = 0.0;
    _v->user_data = NULL;
    return _v;
}

void backward_add(value *self)
{
    if (!self->childrens)
        return;

    for (int i = 0; i < self->num_childrens; i++)
    {
        self->childrens[i]->_grad += self->_grad;
    }
}

void backward_mul(value *self)
{
    if (!self->childrens)
        return;
    int n = self->num_childrens;

    /** TODO: still expensive need to make a fast allocator */
    float *prefix = ALLOC(float, n);
    memset(prefix, 0, n * sizeof(float));
    float *suffix = ALLOC(float, n);
    memset(suffix, 0, n * sizeof(float));

    /**
     * 1, 2, 3, 4
     * 1  2  6  24
     * 24 24 12  14    avoiding divison in this case although it is  three pass
     */

    prefix[0] = self->childrens[0]->data;
    for (int i = 1; i < n; i++)
    {
        prefix[i] = prefix[i - 1] * self->childrens[i]->data;
    }

    suffix[n - 1] = self->childrens[n - 1]->data;
    for (int i = n - 2; i >= 0; i--)
    {
        suffix[i] = suffix[i + 1] * self->childrens[i]->data;
    }

    for (int i = 0; i < n; i++)
    {
        float grad_c = 1.0f;
        if (i > 0)
            grad_c *= prefix[i - 1];
        if (i < n - 1)
            grad_c *= suffix[i + 1];
        self->childrens[i]->_grad += self->_grad * grad_c;
    }

    DEALLOC(prefix);
    DEALLOC(suffix);
}

void backward_div(value *self)
{
    if (!self->childrens)
        return;
    int n = self->num_childrens;
    float deno_ = 1.0f;

    value **c = self->childrens; // caching for pointer redundancy indeirection
    for (int i = 1; i < n; i++)
    {
        deno_ *= c[i]->data;
    }

    c[0]->_grad += self->_grad / deno_;

    for (int i = 1; i < n; i++)
    {
        c[i]->_grad -= (self->_grad * c[0]->data) / (deno_ * c[i]->data);
    }
}

void backward_sub(value *self)
{
    if (!self->childrens)
        return;

    int n = self->num_childrens;
    float d = self->_grad;
    value **a = self->childrens;

    if (n > 0)
    {
        a[0]->_grad += d;

        for (int i = 1; i < n; i++)
        {
            a[i]->_grad -= d;
        }
    }
}

void backward_pow_(value* self) {
    if (!self->childrens) return;
    value* base = self->childrens[0];
    float _d    = base->data;
    float exp   = *(float*)self->user_data;  /*TODO: Needs a Fix so much pointer indirectin bro */

    /** nx^n-1 */
    base->_grad += self->_grad * exp * powf(_d, exp - 1);
}

value *n_add(int n, ...)
{
    va_list args;
    va_start(args, n);

    float sum = 0.0f;
    value **chidldren = NULL;
    if (GLOBAL_TAPE.require_grad)
        chidldren = ALLOC(value *, n);

    for (int i = 0; i < n; i++)
    {
        value *v = va_arg(args, value *);
        sum += v->data;
        if (GLOBAL_TAPE.require_grad)
            chidldren[i] = v;
    }

    va_end(args);

    value *result = value_init(sum);
    if (GLOBAL_TAPE.require_grad)
    {
        result->childrens = chidldren;
        result->num_childrens = n;
        result->bk = backward_add;
        da_append(&GLOBAL_TAPE, result);
    }
    return result;
}

value *n_mul(int n, ...)
{
    va_list args;
    va_start(args, n);
    float m_ = 1.0f;

    value **children = NULL;

    if (GLOBAL_TAPE.require_grad)
    {
        children = ALLOC(value *, n);
    }

    for (int i = 0; i < n; i++)
    {
        value *v = va_arg(args, value *);
        m_ *= v->data;
        if (GLOBAL_TAPE.require_grad)
            children[i] = v;
    }
    va_end(args);

    value *result = value_init(m_);

    if (GLOBAL_TAPE.require_grad)
    {
        result->childrens = children;
        result->num_childrens = n;
        result->bk = backward_mul;
        da_append(&GLOBAL_TAPE, result);
    }

    return result;
}

value *n_div(int n, ...)
{
    va_list args;

    va_start(args, n);
    value *d_ = va_arg(args, value *);
    float q = d_->data;
    value **children = NULL;

    if (GLOBAL_TAPE.require_grad)
    {
        children = ALLOC(value *, n);
        children[0] = d_;
    }

    for (int i = 1; i < n; i++)
    {
        value *v = va_arg(args, value *);
        q = q / v->data;
        if (GLOBAL_TAPE.require_grad)
            children[i] = v;
    }
    va_end(args);

    value *result = value_init(q);

    if (GLOBAL_TAPE.require_grad)
    {
        result->num_childrens = n;
        result->childrens = children;
        result->bk = backward_div;
        da_append(&GLOBAL_TAPE, result);
    }

    return result;
}

value *n_sub(int n, ...)
{
    va_list args;
    va_start(args, n);

    value *_d = va_arg(args, value *);
    float d_ = _d->data;
    value **children = NULL;

    if (GLOBAL_TAPE.require_grad)
    {
        children = ALLOC(value *, n);
        children[0] = _d;
    }

    for (int i = 1; i < n; i++)
    {
        value *v = va_arg(args, value *);
        d_ -= v->data;
        if (GLOBAL_TAPE.require_grad)
            children[i] = v;
    }
    va_end(args);

    value *result = value_init(d_);
    if (GLOBAL_TAPE.require_grad)
    {
        result->childrens = children;
        result->bk = backward_sub;
        result->num_childrens = n;
        da_append(&GLOBAL_TAPE, result);
    }
    return result;
}

value *n_pow_(value *a, float b)
{
    float _da = powf(a->data, b);
    value *res = value_init(_da);

    float *d = ALLOC(float, 1);
    *d = b;

    res->user_data = (void *)d;

    /** Please CPU predict the fking branch */
    if (GLOBAL_TAPE.require_grad)
    {
        res->childrens = ALLOC(value *, 1);
        res->childrens[0] = a;
        res->num_childrens = 1;
        res->bk = backward_pow_;
        da_append(&GLOBAL_TAPE, res);
    }

    return res;
}

void n_backward()
{
    value **arr = GLOBAL_TAPE.items;
    int sz = GLOBAL_TAPE.count;

    for (int i = sz - 1; i >= 0; i--)
    {
        arr[i]->bk(arr[i]);
    }
}
