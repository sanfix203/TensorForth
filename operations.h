#ifndef OPERATIONS_H
#define OPREATIONS_H

#include "tensor.h"
#include "stack.h"

Tensor* tensor_add(Tensor *a, Tensor *b);
Tensor* tensor_sub(Tensor *a, Tensor *b);
Tensor* tensor_mul(Tensor *a, Tensor *b);

Tensor* tensor_greater(Tensor *a, Tensor *b);
Tensor* tensor_less(Tensor *a, Tensor *b);
Tensor* tensor_equal(Tensor *a, Tensor *b);

Tensor* tensor_and(Tensor *a, Tensor *b);
Tensor* tensor_or(Tensor *a, Tensor *b);
Tensor* tensor_not(Tensor *a);

Tensor* tensor_select(Tensor* m, Tensor* a, Tensor* b);

Tensor* tensor_matrix_prod(Tensor* a, Tensor* b);

Tensor* tensor_reshape(Tensor* a, Tensor* s);
Tensor* tensor_ravel(Tensor* t);
Tensor* tensor_shape(Tensor* t);

void tensor_print(Tensor *t);

#endif