#ifndef OPERATIONS_H
#define OPREATIONS_H

#include "tensor.h"
#include "stack.h"
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

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
Tensor* tensor_dot_prod(Tensor* a, Tensor* b);
Tensor* tensor_conv2d(Tensor* a, Tensor* k);

Tensor* tensor_reshape(Tensor* a, Tensor* s);
Tensor* tensor_ravel(Tensor* t);
Tensor* tensor_shape(Tensor* t);

Tensor* tensor_random(Tensor* s);

Tensor* tensor_relu(Tensor* t);
Tensor* tensor_min(Tensor* a, Tensor* b);
Tensor* tensor_max(Tensor* a, Tensor* b);

Tensor* tensor_sum(Tensor* t);

Tensor* tensor_fill(Tensor* s, Tensor* v);

Tensor* tensor_duplicate(Tensor* t);

void skip_pgm_comments(FILE *f);
void tensor_save_pgm(Tensor* t, const char* filename);
Tensor* tensor_load_pgm(const char* filename);
Tensor* tensor_load_pgm(const char* filename);

void tensor_save_raw(Tensor* t, const char* filename);
Tensor* tensor_load_raw(const char* filename);

void tensor_print(Tensor *t);

#endif