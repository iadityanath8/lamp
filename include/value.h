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

#ifndef VALUE_H
#define VALUE_H

#include "utils.h"
#include <stdlib.h>
#include <stdbool.h>

#define COUNT_ARGS(...) (sizeof((value *[]){__VA_ARGS__}) / sizeof(value *))
#define V_ADD(...) n_add(COUNT_ARGS(__VA_ARGS__), __VA_ARGS__)
#define V_MUL(...) n_mul(COUNT_ARGS(__VA_ARGS__), __VA_ARGS__)
#define V_DIV(...) n_div(COUNT_ARGS(__VA_ARGS__), __VA_ARGS__)
#define V_SUB(...) n_sub(COUNT_ARGS(__VA_ARGS__), __VA_ARGS__)
#define V_POW_(V, p) n_pow_(V, p)

typedef struct value value;
typedef struct tape tape;
typedef void (*backwarfFn)(value *);

struct value
{

    /** Main Data used for storing the values in float32 format 
     *  as it is 32 bit and simd avx can store 8 of these in a avx2 and 16 
     *  of these in avx512 based system so float is preferred in this for high 
     *  precision double of f64 is recommended */
    float data;
    
    float _grad;

    /** takes a value and calculates the backward of the immediate 
     *  value stored */
    backwarfFn bk;

    /** Array of dependencies  used for propogating the gradients on
     *  a computational graph */
    _Many(value*) childrens;
    
    /** user data as we dont have
     *  lamdas / closures in C **/
    size_t num_childrens;
    

    /** User data like we dont have any lamdas in this so if a custome routine is 
     *  written this can be used to pass the information to the routine through this 
     *  opaque type  */
    void* user_data;
};

struct tape
{
    /** this will contain the pointer to the value as an array value in here  **/
    value **items;  

    /** this uses da_append symantics in here check
     *  for reference for dynamic array by Alex Kutepoy
     *  ----------------------------------------------------
     *  https://github.com/tsoding/nob.h/blob/main/nob.h  */
    size_t count;
    size_t capacity;
    bool require_grad;
};


value *value_init(float _d);
value *n_add(int n, ...);

/**
 *      a = b * c * e * f
 *
 *      da / db = c*e*f
 *      db /  dc = b* e*f
    */

value *n_mul(int n, ...);
value *n_div(int n, ...);
value *n_sub(int n, ...);

/** under construction  */
value *n_pow(int n, ...);
/** under construction */

value *n_pow_(value* a, float b);

void n_backward();

/** Tape initialization routines */

extern tape GLOBAL_TAPE; 
void tape_init();
void tape_deinit();
void dump_tape();
#endif