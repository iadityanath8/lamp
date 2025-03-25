/*
    Basic exmaple using value type for automatic gradient calculation 
    for now link with -lm math library and then also ../src/value.c 
    the c implementation of this header soon we are planning to make a 
    dll and so out of it and also the static linked file 
*/
#include "../include/value.h"
#include <stdio.h>

/**
 * x = 2
 * y = 3
 * c = a * b
 * z = x * y
 * 
 * dz/ dx = 1 * 3
 * dz / dy = 2 * 1
 * dc / da = 
 * dc / db = 0+ 1 
 * 
 * 
 */

int main() {
    tape_init();
    value* x = value_init(2);
    value* y = value_init(3);
    
    // value* z = V_POW_(x, 2); // x^3 => 3x^2  // chain rule  d = (x + y)^2 + z
    value* z = V_ADD(V_POW_(x, 2), V_POW_(y, 2));
    
    z->_grad = 1.0f;

    n_backward();
    printf("%f %f\n",x->_grad, y->_grad);
    tape_deinit();
    return  0;
}
