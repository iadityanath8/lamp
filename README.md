# Lamp

A high-performance numerical computation library written in C, featuring `NDArray`, `Tensor`, and `Value` objects with automatic differentiation. This library is designed for deep learning, scientific computing, and general-purpose numerical operations.

## Features
- **NDArray**: A NumPy-like n-dimensional array for efficient numerical computations with cuBLAS backend support for GPU acceleration.
- **Tensor** (Under Construction): A multi-dimensional data structure with built-in automatic differentiation, currently being developed.
- **Value**: A scalar computation class supporting autograd for deep learning models.
- **Optimized Backend**: Uses Intel MKL and cuBLAS for high-performance computations.
- **Custom Preprocessor**: Enables template-based programming for high performance.

## Installation
```sh
# Clone the repository
git clone https://github.com/yourusername/yourlibrary.git
cd yourlibrary

# Build the library
make
```

## Usage

### NDArray Example
```c
#include "ndarray.h"
#include <stdio.h>

int main() {
    NDArray *a = ndarray_init_with_arr(SHP(2,2),2,ARR(1,2,3,4));
    ndarray_put(a, 0, 0, 1.0);
    ndarray_put(a, 0, 1, 2.0);
    ndarray_free(a);
    return 0;
}
```

### Tensor Example (Under Construction)
```c
#include "tensor.h"
#include <stdio.h>

int main() {
    tensor* a = tensor_init(ARR(1, 2, 3,5), SHP(2, 2),2); // create a 2x2 matrix 
    tensor* b = tensor_init(ARR(1, 2, 3,5), SHP(2, 2),2); // create a 2x2 matrix 
    
    tensor* c = T_ADD(a, b);   // macro can take variable number of arguments 
    return 0;
}
```

### Value Object Example
```c
/*
    Basic example using Value type for automatic gradient calculation. 
    For now, link with the -lm math library and ../src/value.c 
    The C implementation of this header. Soon, we plan to make a 
    shared library (DLL) and a statically linked file.
*/
#include "value.h"
#include <stdio.h>

int main() {
    tape_init();
    value* x = value_init(2);
    value* y = value_init(3);
    
    // z = x^2 + y^2
    value* z = V_ADD(V_POW_(x, 2), V_POW_(y, 2));
    
    z->_grad = 1.0f;
    n_backward();
    
    printf("Gradient of x: %f\n", x->_grad);
    printf("Gradient of y: %f\n", y->_grad);
    
    tape_deinit();
    return  0;
}
```

## Roadmap

### ✅ Completed
- **NDArray Implementation**: Added cuBLAS backend for GPU acceleration.
- **Value Object with Automatic Differentiation**: Supports scalar autograd operations.

### 🚧 In Progress
- **Tensor Operations (Under Construction)**: Implementing a multi-dimensional Tensor structure with efficient operations.
- **GPU Acceleration for Tensor**: Adding cuBLAS support for matrix operations in `Tensor`.
- **Optimization and Performance Tuning**: Improving execution speed and memory efficiency.

### 🔥 Upcoming Features
- **Gradient-Based Optimizers**: Implement Adam, SGD, and RMSprop for model training.
- **Autograd Enhancements**: Extend `Value` and `Tensor` to support complex computational graphs.
- **Custom Preprocessor Wrapper Compiler**: Enable template-based programming for high performance.
- **Shared Library (DLL) and Static Library**: Provide easy linking for integration into other projects.
- **Integration with Other Libraries**: OpenBlas MKL for only CPU optimized performance also cuda/ Cublas for GPU.

## License
GNUv3 License. See `LICENSE` for details.
