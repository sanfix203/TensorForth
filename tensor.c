// Gabriele Sanfilippo SM3201618

#include "tensor.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/mman.h>

size_t tensor_get_num_elements(int32_t n_dim, const int32_t *shape) {
    if (n_dim <= 0 || n_dim > MAX_DIM) 
        fprintf(stderr, "Errore [tensor_get_num_elements]: n_dim (%d) non valido. Deve essere compreso tra 1 e %d.\n", n_dim, MAX_DIM);
    
    size_t total_elements = 1;
    for (int i = 0; i < n_dim; i++) total_elements *= shape[i];
    return total_elements;
}

Tensor* tensor_create(int32_t n_dim, int32_t *shape) {
    // Controllo dimensioni
    if (n_dim <= 0 || n_dim > MAX_DIM || shape == NULL) {
        fprintf(stderr, "Errore: Dimensioni non valide per la creazione del tensore.\n");
        return NULL;
    }

    // Allocazione del tensore e del tensor_buffer dati
    Tensor *t = (Tensor*)malloc(sizeof(Tensor));
    if (t == NULL) {
        perror("Errore di allocazione memoria per Tensor");
        exit(EXIT_FAILURE);
    }

    TensorBuffer *buf = (TensorBuffer*)malloc(sizeof(TensorBuffer));
    if (buf == NULL) {
        perror("Errore di allocazione memoria per TensorBuffer");
        free(t);
        exit(EXIT_FAILURE);
    }

    // Allocazioni elementi
    size_t num_elements = tensor_get_num_elements(n_dim, shape);
    buf->data = (float*)malloc(num_elements * sizeof(float));
    if (buf->data == NULL) {
        perror("Errore di allocazione memoria per i dati del tensore");
        free(buf);
        free(t);
        exit(EXIT_FAILURE);
    }

    // Inizializzazione valori
    buf->ref_count = 1; 
    buf->mmap_ptr = NULL; // <-- AGGIUNGI QUESTO
    buf->mmap_size = 0;   // <-- AGGIUNGI QUESTO

    t->n_dim = n_dim;
    t->tensor_buffer = buf;
    
    for (int i = 0; i < n_dim; i++) {
        t->shape[i] = shape[i];
    }
    for (int i = n_dim; i < MAX_DIM; i++) {
        t->shape[i] = 0;
    }

    return t;
}

Tensor* tensor_reference(Tensor *t) {
    if (t == NULL) return NULL;

    Tensor *new_t = (Tensor*)malloc(sizeof(Tensor));
    if (new_t == NULL) {
        perror("Errore di allocazione memoria per Tensor (reference)");
        exit(EXIT_FAILURE);
    }

    new_t->n_dim = t->n_dim;
    for (int i = 0; i < MAX_DIM; i++) {
        new_t->shape[i] = t->shape[i];
    }

    new_t->tensor_buffer = t->tensor_buffer;
    new_t->tensor_buffer->ref_count++;

    return new_t;
}

void tensor_free(Tensor *t) {
    if (t == NULL) return;

    t->tensor_buffer->ref_count--;

    if (t->tensor_buffer->ref_count == 0) {
            if (t->tensor_buffer->mmap_ptr != NULL) {
                munmap(t->tensor_buffer->mmap_ptr, t->tensor_buffer->mmap_size);
            } 
            else if (t->tensor_buffer->data != NULL) {
                free(t->tensor_buffer->data);
            }
            free(t->tensor_buffer);
        }
    
    free(t);
}