// Gabriele Sanfilippo SM3201618

#ifndef TENSOR_H
#define TENSOR_H

// ? #define MAX_SIZE 10000
#define MAX_DIM 2

#include <stdint.h>
#include <stdio.h>
#include <sys/types.h>

typedef struct {
    float *data;
    int ref_count;
} TensorBuffer;

typedef struct {
    TensorBuffer *tensor_buffer; // dopo il reshape il tensore punterà alla stessa tensordata
    int32_t shape[MAX_DIM]; // lunghezza, un valore per vettori e due valori per matrici
    int32_t n_dim; // numero di dimensioni
} Tensor;

// Creazione tensore e aumento dei reference
Tensor* create (int32_t dim, int32_t shape);

// aggiornamento numero di riferimenti al tensore
Tensor* tensor_reference(Tensor *t);

// Eliminazione di riferimento. Se il numero di riferimenti
// raggiunge 0, i dati vengono deallocati
void tensor_free(Tensor *t);

size_t tensor_get_num_elements(int32_t ndim, const int32_t *shape);

#endif