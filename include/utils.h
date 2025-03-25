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

#ifndef UTILS_H
#define UTILS_H

#include <assert.h>

#define DA_INIT_CAP 256

#ifdef N_ALLOC
#   error "Allocator in Production !!! use DO Not activate the N_ALLOC flag now use default ALLOC"
#else
#   define ALLOC(T, ele) (T *)malloc(sizeof(T) * ele)
#   define DEALLOC(ptr) free(ptr)
#   define REALLOC realloc
#endif

#define ASSERT assert
#define _Many(ptr) ptr*

/** TODO: Change of name is required */
#define SHP(...) (typeof(__VA_ARGS__)[]){__VA_ARGS__}
#define ARR(...) (float[]){__VA_ARGS__}
// #define SHPF(...) (float[]){__VA_ARGS__}

#define CHECK_ALLOC(ptr)                                                                \
    if (!(ptr))                                                                         \
    {                                                                                   \
        fprintf(stderr, "Memory allocation failed: %s, line %d\n", __FILE__, __LINE__); \
        exit(EXIT_FAILURE);                                                             \
    }

#define ASSERT_SAME_SHAPE(A, B)                     \
    do                                              \
    {                                               \
        assert((A)->dim == (B)->dim);               \
        for (size_t i = 0; i < (A)->dim; i++)       \
        {                                           \
            assert((A)->shape[i] == (B)->shape[i]); \
        }                                           \
    } while (0)

#define da_reserve(da, expected_capacity)                                              \
    do                                                                                 \
    {                                                                                  \
        if ((expected_capacity) > (da)->capacity)                                      \
        {                                                                              \
            if ((da)->capacity == 0)                                                   \
            {                                                                          \
                (da)->capacity = DA_INIT_CAP;                                          \
            }                                                                          \
            while ((expected_capacity) > (da)->capacity)                               \
            {                                                                          \
                (da)->capacity *= 2;                                                   \
            }                                                                          \
            (da)->items = REALLOC((da)->items, (da)->capacity * sizeof(*(da)->items)); \
            ASSERT((da)->items != NULL && "Buy more RAM lol");                         \
        }                                                                              \
    } while (0)

// Append an item to a dynamic array
/** Taken from tsodings dynamic array: -> https://github.com/tsoding/nob.h/blob/main/nob.h */
#define da_append(da, item)                  \
    do                                       \
    {                                        \
        da_reserve((da), (da)->count + 1);   \
        (da)->items[(da)->count++] = (item); \
    } while (0)
#endif