// Gabriele Sanfilippo SM3201618

#include "operations.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h> // per operazioni di comparazione con float

// Operazioni aritmetiche
Tensor* tensor_add(Tensor *a, Tensor *b){
    // Controllo
    if (a == NULL || b == NULL) return NULL;
    if (a->n_dim != b->n_dim) {
        fprintf(stderr, "Errore di esecuzione: Impossibile eseguire l'operazione con tensori con n_dim diverso.\n");
        exit(EXIT_FAILURE);
    }
    
    for (int i = 0; i < a->n_dim; i++) {
        if (a->shape[i] != b->shape[i]) {
            fprintf(stderr, "Errore di esecuzione: Impossibile eseguire l'operazione con tensori con shape diversa.\n");
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
        fprintf(stderr, "Errore di esecuzione: Impossibile eseguire l'operazione tensori con n_dim diverso.\n");
        exit(EXIT_FAILURE);
    }
    
    for (int i = 0; i < a->n_dim; i++) {
        if (a->shape[i] != b->shape[i]) {
            fprintf(stderr, "Errore di esecuzione: Impossibile eseguire l'operazione tensori con shape diversa.\n");
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
        fprintf(stderr, "Errore di esecuzione: Impossibile eseguire l'operazione con tensori con n_dim diverso.\n");
        exit(EXIT_FAILURE);
    }
    
    for (int i = 0; i < a->n_dim; i++) {
        if (a->shape[i] != b->shape[i]) {
            fprintf(stderr, "Errore di esecuzione: Impossibile eseguire l'operazione con tensori con shape diversa.\n");
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

// Operazioni di comparazione
Tensor* tensor_greater(Tensor *a, Tensor* b){
    // Controllo
    if (a == NULL || b == NULL) return NULL;
    if (a->n_dim != b->n_dim) {
        fprintf(stderr, "Errore di esecuzione: Impossibile eseguire l'operazione con tensori con n_dim diverso.\n");
        exit(EXIT_FAILURE);
    }
    
    for (int i = 0; i < a->n_dim; i++) {
        if (a->shape[i] != b->shape[i]) {
            fprintf(stderr, "Errore di esecuzione: Impossibile eseguire l'operazione con tensori con shape diversa.\n");
            exit(EXIT_FAILURE);
        }
    }

    // Operazione
    Tensor *result = tensor_create(a->n_dim, a->shape);
    size_t num_elements = tensor_get_num_elements(a->n_dim, a->shape);

    #pragma omp parallel for
    for (size_t i = 0; i < num_elements; i++) {
        result->tensor_buffer->data[i] = (a->tensor_buffer->data[i] > b->tensor_buffer->data[i]) ? 1 : 0;
    }

    return result;
}
Tensor* tensor_less(Tensor *a, Tensor* b){
    // Controllo
    if (a == NULL || b == NULL) return NULL;
    if (a->n_dim != b->n_dim) {
        fprintf(stderr, "Errore di esecuzione: Impossibile eseguire l'operazione con  tensori con n_dim diverso.\n");
        exit(EXIT_FAILURE);
    }
    
    for (int i = 0; i < a->n_dim; i++) {
        if (a->shape[i] != b->shape[i]) {
            fprintf(stderr, "Errore di esecuzione: Impossibile eseguire l'operazione con tensori con shape diversa.\n");
            exit(EXIT_FAILURE);
        }
    }

    // Operazione
    Tensor *result = tensor_create(a->n_dim, a->shape);
    size_t num_elements = tensor_get_num_elements(a->n_dim, a->shape);

    #pragma omp parallel for
    for (size_t i = 0; i < num_elements; i++) {
        result->tensor_buffer->data[i] = (a->tensor_buffer->data[i] < b->tensor_buffer->data[i]) ? 1 : 0;
    }

    return result;
}
Tensor* tensor_equal(Tensor *a, Tensor* b){
    // Controllo
    if (a == NULL || b == NULL) return NULL;
    if (a->n_dim != b->n_dim) {
        fprintf(stderr, "Errore di esecuzione: Impossibile eseguire l'operazione tensori con n_dim diverso.\n");
        exit(EXIT_FAILURE);
    }
    
    for (int i = 0; i < a->n_dim; i++) {
        if (a->shape[i] != b->shape[i]) {
            fprintf(stderr, "Errore di esecuzione: Impossibile eseguire l'operazione tensori con shape diversa.\n");
            exit(EXIT_FAILURE);
        }
    }

    // Operazione
    Tensor *result = tensor_create(a->n_dim, a->shape);
    size_t num_elements = tensor_get_num_elements(a->n_dim, a->shape);

    #pragma omp parallel for
    for (size_t i = 0; i < num_elements; i++) {
        float diff = fabsf(a->tensor_buffer->data[i] - b->tensor_buffer->data[i]);
        result->tensor_buffer->data[i] = (diff < 1e-6) ? 1 : 0;
    }

    return result;
}

// Operazioni logiche
static void check_boolean_tensor(Tensor *t, size_t num_elements) {
    if (t == NULL) return;
    
    for (size_t i = 0; i < num_elements; i++) {
        float val = t->tensor_buffer->data[i];
        if (val != 0.0f && val != 1.0f) {
            fprintf(stderr, "Errore di esecuzione: Le operazioni logiche richiedono tensori composti solo da 0.0 e 1.0. Trovato: %f\n", val);
            exit(EXIT_FAILURE);
        }
    }
}
Tensor* tensor_and(Tensor *a, Tensor *b){
    // Controllo
    if (a == NULL || b == NULL) return NULL;
    if (a->n_dim != b->n_dim) {
        fprintf(stderr, "Errore di esecuzione: Impossibile eseguire l'operazione con tensori con n_dim diverso.\n");
        exit(EXIT_FAILURE);
    }
    
    for (int i = 0; i < a->n_dim; i++) {
        if (a->shape[i] != b->shape[i]) {
            fprintf(stderr, "Errore di esecuzione: Impossibile eseguire l'operazione con tensori con shape diversa.\n");
            exit(EXIT_FAILURE);
        }
    }

    size_t num_elements = tensor_get_num_elements(a->n_dim, a->shape);
    check_boolean_tensor(a, num_elements);
    check_boolean_tensor(b, num_elements);

    // Operazione
    Tensor *result = tensor_create(a->n_dim, a->shape);

    #pragma omp parallel for
    for (size_t i = 0; i < num_elements; i++) {
        result->tensor_buffer->data[i] = (a->tensor_buffer->data[i] == 1.0f && b->tensor_buffer->data[i] == 1.0f) ? 1.0f : 0.0f;
    }

    return result;
}
Tensor* tensor_or(Tensor *a, Tensor *b){
    // Controllo
    if (a == NULL || b == NULL) return NULL;
    if (a->n_dim != b->n_dim) {
        fprintf(stderr, "Errore di esecuzione: Impossibile eseguire l'operazione con tensori con n_dim diverso.\n");
        exit(EXIT_FAILURE);
    }
    
    for (int i = 0; i < a->n_dim; i++) {
        if (a->shape[i] != b->shape[i]) {
            fprintf(stderr, "Errore di esecuzione: Impossibile eseguire l'operazione con tensori con shape diversa.\n");
            exit(EXIT_FAILURE);
        }
    }

    size_t num_elements = tensor_get_num_elements(a->n_dim, a->shape);
    check_boolean_tensor(a, num_elements);
    check_boolean_tensor(b, num_elements);

    // Operazione
    Tensor *result = tensor_create(a->n_dim, a->shape);

    #pragma omp parallel for
    for (size_t i = 0; i < num_elements; i++) {
        result->tensor_buffer->data[i] = (a->tensor_buffer->data[i] == 1.0f || b->tensor_buffer->data[i] == 1.0f) ? 1.0f : 0.0f;
    }

    return result;
}
Tensor* tensor_not(Tensor *a){
    // Controllo
    if (a == NULL) return NULL;

    size_t num_elements = tensor_get_num_elements(a->n_dim, a->shape);
    check_boolean_tensor(a, num_elements);

    // Operazione
    Tensor *result = tensor_create(a->n_dim, a->shape);

    #pragma omp parallel for
    for (size_t i = 0; i < num_elements; i++) {
        result->tensor_buffer->data[i] = (a->tensor_buffer->data[i] == 1.0f) ? 0.0f : 1.0f;
    }

    return result;
}



void tensor_print(Tensor *t){
    // Controllo
    if (t==NULL) return;

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

