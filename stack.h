// Gabriele Sanfilippo SM3201618

#ifndef STACK_H
#define STACK_H

#define STACK_MAX_SIZE 1024

#include "tensor.h"

typedef struct {
    Tensor* items[STACK_MAX_SIZE];
    int top;
} Stack;



#endif