// Gabriele Sanfilippo SM3201618

#include "stack.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void stack_init(Stack *s) {
    if (s == NULL) return;
    s->top = 0;
}

void stack_push_tensor(Stack *s, Tensor *t) {
    if (s == NULL) return;
    if (s->top >= STACK_MAX_SIZE) {
        fprintf(stderr, "Errore: Stack Overflow (Limite massimo: %d)\n", STACK_MAX_SIZE);
        exit(EXIT_FAILURE);
    }
    
    s->items[s->top].type = ITEM_TENSOR;
    s->items[s->top].data.tensor = t;
    
    s->top++;
}

void stack_push_string(Stack *s, char *str) {
    if (s == NULL || str == NULL) return;
    if (s->top >= STACK_MAX_SIZE) {
        fprintf(stderr, "Errore: Stack Overflow!\n");
        exit(EXIT_FAILURE);
    }
    
    s->items[s->top].type = ITEM_STRING;
    
    // Sostituiamo strdup con malloc e strcpy (Standard C sicuro)
    s->items[s->top].data.string = (char *)malloc(strlen(str) + 1);
    if (s->items[s->top].data.string == NULL) {
        fprintf(stderr, "Errore: Allocazione memoria fallita per la stringa.\n");
        exit(EXIT_FAILURE);
    }
    strcpy(s->items[s->top].data.string, str);
    
    s->top++;
}

Tensor* stack_pop_tensor(Stack *s) {
    if (s == NULL || s->top <= 0) {
        fprintf(stderr, "Errore: Stack vuoto. Impossibile estrarre operando.\n");
        exit(EXIT_FAILURE);
    }
    
    s->top--;
    StackItem item = s->items[s->top];
    
    if (item.type != ITEM_TENSOR) {
        fprintf(stderr, "Errore di Tipo: Atteso un Tensore, trovato un Nome File.\n");
        exit(EXIT_FAILURE);
    }
    
    return item.data.tensor;
}

char* stack_pop_string(Stack *s) {
    if (s == NULL || s->top <= 0) {
        fprintf(stderr, "Errore: Stack vuoto. Impossibile estrarre stringa.\n");
        exit(EXIT_FAILURE);
    }
    
    s->top--;
    StackItem item = s->items[s->top];
    
    if (item.type != ITEM_STRING) {
        fprintf(stderr, "Errore di Tipo: Atteso un Nome File, trovato un Tensore.\n");
        exit(EXIT_FAILURE);
    }
    
    return item.data.string;
}

StackItem stack_pop(Stack *s) {
    if (s->top <= 0) { 
        fprintf(stderr, "Errore: Stack vuoto. Impossibile estrarre stringa.\n");
        exit(EXIT_FAILURE); }

    s->top--;
    return s->items[s->top];
}

Tensor* stack_peek_tensor(Stack *s) {
    if (s == NULL || s->top <= 0) {
        fprintf(stderr, "Errore: Stack vuoto (peek).\n");
        exit(EXIT_FAILURE);
    }
    
    StackItem item = s->items[s->top - 1];
    if (item.type != ITEM_TENSOR) {
        fprintf(stderr, "Errore di Tipo: Peek attende un Tensore.\n");
        exit(EXIT_FAILURE);
    }
    
    return item.data.tensor;
}

void stack_cleanup(Stack *s) {
    if (s == NULL) return;
    
    while (s->top > 0) {
        s->top--;
        StackItem item = s->items[s->top];
        
        if (item.type == ITEM_TENSOR) {
            tensor_free(item.data.tensor);
        } else if (item.type == ITEM_STRING) {
            free(item.data.string); 
        }
    }
}