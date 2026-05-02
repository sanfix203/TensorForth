// Gabriele Sanfilippo SM3201618

#ifndef STACK_H
#define STACK_H

#include "tensor.h"
#include <stdbool.h>

#define STACK_MAX_SIZE 1024

typedef enum {
    ITEM_TENSOR,
    ITEM_STRING
} ItemType;

typedef struct {
    ItemType type;
    union {
        Tensor *tensor; // Usato se type == ITEM_TENSOR
        char *string;   // Usato se type == ITEM_STRING
    } data;
} StackItem;

/* Seppur occupare un intera sezione di memoria proporzionale a STACK_MAX_SIZE
/ possa sembrare uno spreco, valutando i trade-off tra allocarla subito e allocarla
/ dinamicamente con malloc si è scelta la prima opzione per aumentare la velocità
/ di esecuzione in cambio di un po' di spazio occupato in più */
typedef struct {
    StackItem items[STACK_MAX_SIZE];
    int top;
} Stack;

void stack_init(Stack *s);
void stack_cleanup(Stack *s);

void stack_push_tensor(Stack *s, Tensor *t);
void stack_push_string(Stack *s, const char *str);

Tensor* stack_pop_tensor(Stack *s);
char* stack_pop_string(Stack *s);

Tensor* stack_peek_tensor(Stack *s);

#endif // STACK_H