// Gabriele Sanfilippo SM3201618

#include "stack.h"
#include <stdio.h>
#include <stdlib.h>

void stack_init(Stack *s) {
    if (s == NULL){
        fprintf(stderr, "Impossibile inizializzare lo stack");
        exit(EXIT_FAILURE);
    };
    s->top = 0;
}

void stack_push(Stack *s, Tensor *t) {
    if (s == NULL) return;
    
    if (s->top >= STACK_MAX_SIZE) {
        fprintf(stderr, "Errore Fatale: Stack Overflow.\n");
        exit(EXIT_FAILURE);
    }
    
    s->items[s->top] = t;
    s->top++;
}

Tensor* stack_pop(Stack *s) {
    if (s == NULL) return NULL;
    
    if (s->top <= 0) {
        fprintf(stderr, "Errore di esecuzione: Stack Underflow.\n");
        exit(EXIT_FAILURE); 
    }
    
    s->top--;
    return s->items[s->top];
}

Tensor* stack_peek(Stack *s) {
    if (s == NULL || s->top <= 0) {
        fprintf(stderr, "Errore di esecuzione: Stack vuoto (peek).\n");
        exit(EXIT_FAILURE);
    }
    return s->items[s->top - 1];
}

void stack_cleanup(Stack *s) {
    if (s == NULL) return;
    
    while (s->top > 0) {
        Tensor *t = stack_pop(s);
        tensor_free(t);
    }
}