#include "../include/tensor.h"
#include <stdbool.h>

void printb(ndarray* a){
  int ROW = a->shape[0];
  int COL = a->shape[1];

  for (int i =0; i < ROW;i++) {
    for (int j = 0; j < COL;j++) {
      printf("%f ",ndarray_get_element(a,SHP(i,j)));
    }
    printf("\n");
  }
}

int main() {
  float a[] = {
    1,2,
    3,4
  };

  float b[] = {
    4,5,
    6,7
  };
  
  tensor* a_ = tensor_init_with_arr(a,SHP(2,2),2);
  tensor* b_ = tensor_init_with_arr(b,SHP(2,2),2);
  tensor* c  = t_matmul(a_,b_);
  c->grad = ndarray_ones(c->data->shape,c->data->dim);

  c->bk(c);
  
  /** Manuall */
  ndarray_dump(b_->grad);
  return 0;
  // c_t->bk(c_t);
}