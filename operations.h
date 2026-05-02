#ifndef OPERATIONS_H
#define OPREATIONS_H

#include "tensor.h"
#include "stack.h"

Tensor* tensor_add(Tensor *a, Tensor *b);
Tensor* tensor_sub(Tensor *a, Tensor *b);
Tensor* tensor_mul(Tensor *a, Tensor *b);

void tensor_print(Tensor *t);

#endif