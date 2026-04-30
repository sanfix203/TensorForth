// Gabriele Sanfilippo SM3201618

#ifndef STACK_H
#define STACK_H

#define STACK_MAX_SIZE 1024

#include "tensor.h"

typedef struct {
    Tensor* items[STACK_MAX_SIZE];
    int top;
} Stack;

void stack_init(Stack *s);
void stack_push(Stack *s, Tensor *t);
Tensor* stack_pop(Stack *s);
Tensor* stack_peek(Stack *s);
void stack_cleanup(Stack *s);

#endif