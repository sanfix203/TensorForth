// Gabriele Sanfilippo SM3201618

#include "operations.h"
#include <stdio.h>
#include <stdlib.h>

Tensor* tensor_add(Tensor *a, Tensor *b){
    // Controllo
    if (a == NULL || b == NULL) return NULL;
    if (a->n_dim != b->n_dim) {
        fprintf(stderr, "Errore di esecuzione: Impossibile sommare tensori con n_dim diverso.\n");
        exit(EXIT_FAILURE);
    }
    
    for (int i = 0; i < a->n_dim; i++) {
        if (a->shape[i] != b->shape[i]) {
            fprintf(stderr, "Errore di esecuzione: Impossibile sommare tensori con shape diversa.\n");
            exit(EXIT_FAILURE);
        }
    }

    // Operazione
    Tensor *result = tensor_create(a->n_dim, a->shape);
    size_t num_elements = tensor_get_num_elements(a->n_dim, a->shape);

    #pragma omp parallel for
    for (size_t i = 0; i < num_elements; i++) {
        result->tensor_buffer->data[i] = a->tensor_buffer->data[i] + b->tensor_buffer->data[i];
    }

    return result;
}

Tensor* tensor_sub(Tensor *a, Tensor *b){
    // Controllo
    if (a == NULL || b == NULL) return NULL;
    if (a->n_dim != b->n_dim) {
        fprintf(stderr, "Errore di esecuzione: Impossibile sommare tensori con n_dim diverso.\n");
        exit(EXIT_FAILURE);
    }
    
    for (int i = 0; i < a->n_dim; i++) {
        if (a->shape[i] != b->shape[i]) {
            fprintf(stderr, "Errore di esecuzione: Impossibile sommare tensori con shape diversa.\n");
            exit(EXIT_FAILURE);
        }
    }

    // Operazione
    Tensor *result = tensor_create(a->n_dim, a->shape);
    size_t num_elements = tensor_get_num_elements(a->n_dim, a->shape);

    #pragma omp parallel for
    for (size_t i = 0; i < num_elements; i++) {
        result->tensor_buffer->data[i] = a->tensor_buffer->data[i] - b->tensor_buffer->data[i];
    }

    return result;
}

Tensor* tensor_mul(Tensor *a, Tensor *b){
    // Controllo
    if (a == NULL || b == NULL) return NULL;
    if (a->n_dim != b->n_dim) {
        fprintf(stderr, "Errore di esecuzione: Impossibile sommare tensori con n_dim diverso.\n");
        exit(EXIT_FAILURE);
    }
    
    for (int i = 0; i < a->n_dim; i++) {
        if (a->shape[i] != b->shape[i]) {
            fprintf(stderr, "Errore di esecuzione: Impossibile sommare tensori con shape diversa.\n");
            exit(EXIT_FAILURE);
        }
    }

    // Operazione
    Tensor *result = tensor_create(a->n_dim, a->shape);
    size_t num_elements = tensor_get_num_elements(a->n_dim, a->shape);

    #pragma omp parallel for
    for (size_t i = 0; i < num_elements; i++) {
        result->tensor_buffer->data[i] = a->tensor_buffer->data[i] * b->tensor_buffer->data[i];
    }

    return result;
}

void tensor_print(Tensor *t){
    // Controllo
    if (t==NULL) return NULL;

    // Operazione
    printf("Tensor (shape=[");
    for (int i = 0; i < t->n_dim; i++) {
        printf("%d", t->shape[i]);
        if (i < t->n_dim - 1) printf(" ");
    }
    printf("], data=[");
    
    size_t num_elements = tensor_get_num_elements(t->n_dim, t->shape);
    for (size_t i = 0; i < num_elements; i++) {
        printf("%g", t->tensor_buffer->data[i]);
        if (i < num_elements - 1) printf(" ");
    }
    printf("])\n");
}