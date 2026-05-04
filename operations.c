// Gabriele Sanfilippo SM3201618

#include "operations.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h> // per operazioni di comparazione con float

// Funzioni di supporto
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
static bool tensor_have_same_shape(Tensor *a, Tensor *b) {
    if (a == NULL || b == NULL) return false;
    
    if (a->n_dim != b->n_dim) return false;
    
    for (int i = 0; i < a->n_dim; i++) {
        if (a->shape[i] != b->shape[i]) {
            return false;
        }
    }
    
    return true;
}

// Operazioni aritmetiche
Tensor* tensor_add(Tensor *a, Tensor *b){
    if (a==NULL || b==NULL) return NULL;

    if (!tensor_have_same_shape(a, b)){
        fprintf(stderr, "Errore di esecuzione: Impossibile eseguire l'operazione con tensori con shape diversa.\n");
        exit(EXIT_FAILURE);
    }

    Tensor *result = tensor_create(a->n_dim, a->shape);
    size_t num_elements = tensor_get_num_elements(a->n_dim, a->shape);

    #pragma omp parallel for
    for (size_t i = 0; i < num_elements; i++) {
        result->tensor_buffer->data[i] = a->tensor_buffer->data[i] + b->tensor_buffer->data[i];
    }

    return result;
}
Tensor* tensor_sub(Tensor *a, Tensor *b){
    if (a == NULL || b == NULL) return NULL;
    if (!tensor_have_same_shape(a, b)){
        fprintf(stderr, "Errore di esecuzione: Impossibile eseguire l'operazione con tensori con shape diversa.\n");
        exit(EXIT_FAILURE);
    }

    Tensor *result = tensor_create(a->n_dim, a->shape);
    size_t num_elements = tensor_get_num_elements(a->n_dim, a->shape);

    #pragma omp parallel for
    for (size_t i = 0; i < num_elements; i++) {
        result->tensor_buffer->data[i] = a->tensor_buffer->data[i] - b->tensor_buffer->data[i];
    }

    return result;
}
Tensor* tensor_mul(Tensor *a, Tensor *b){
    if (a == NULL || b == NULL) return NULL;
    if (!tensor_have_same_shape(a, b)){
        fprintf(stderr, "Errore di esecuzione: Impossibile eseguire l'operazione con tensori con shape diversa.\n");
        exit(EXIT_FAILURE);
    }
    
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
    if (a == NULL || b == NULL) return NULL;
    if (!tensor_have_same_shape(a, b)){
        fprintf(stderr, "Errore di esecuzione: Impossibile eseguire l'operazione con tensori con shape diversa.\n");
        exit(EXIT_FAILURE);
    }

    Tensor *result = tensor_create(a->n_dim, a->shape);
    size_t num_elements = tensor_get_num_elements(a->n_dim, a->shape);

    #pragma omp parallel for
    for (size_t i = 0; i < num_elements; i++) {
        result->tensor_buffer->data[i] = (a->tensor_buffer->data[i] > b->tensor_buffer->data[i]) ? 1 : 0;
    }

    return result;
}
Tensor* tensor_less(Tensor *a, Tensor* b){
    if (a == NULL || b == NULL) return NULL;
    if (!tensor_have_same_shape(a, b)){
        fprintf(stderr, "Errore di esecuzione: Impossibile eseguire l'operazione con tensori con shape diversa.\n");
        exit(EXIT_FAILURE);
    }
    
    Tensor *result = tensor_create(a->n_dim, a->shape);
    size_t num_elements = tensor_get_num_elements(a->n_dim, a->shape);

    #pragma omp parallel for
    for (size_t i = 0; i < num_elements; i++) {
        result->tensor_buffer->data[i] = (a->tensor_buffer->data[i] < b->tensor_buffer->data[i]) ? 1 : 0;
    }

    return result;
}
Tensor* tensor_equal(Tensor *a, Tensor* b){
    if (a == NULL || b == NULL) return NULL;
    if (!tensor_have_same_shape(a, b)){
        fprintf(stderr, "Errore di esecuzione: Impossibile eseguire l'operazione con tensori con shape diversa.\n");
        exit(EXIT_FAILURE);
    }

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
Tensor* tensor_and(Tensor *a, Tensor *b){
    if (a == NULL || b == NULL) return NULL;
    if (!tensor_have_same_shape(a, b)){
        fprintf(stderr, "Errore di esecuzione: Impossibile eseguire l'operazione con tensori con shape diversa.\n");
        exit(EXIT_FAILURE);
    }

    size_t num_elements = tensor_get_num_elements(a->n_dim, a->shape);
    check_boolean_tensor(a, num_elements);
    check_boolean_tensor(b, num_elements);

    Tensor *result = tensor_create(a->n_dim, a->shape);

    #pragma omp parallel for
    for (size_t i = 0; i < num_elements; i++) {
        result->tensor_buffer->data[i] = (a->tensor_buffer->data[i] == 1.0f && b->tensor_buffer->data[i] == 1.0f) ? 1.0f : 0.0f;
    }

    return result;
}
Tensor* tensor_or(Tensor *a, Tensor *b){
    if (a == NULL || b == NULL) return NULL;
    if (!tensor_have_same_shape(a, b)){
        fprintf(stderr, "Errore di esecuzione: Impossibile eseguire l'operazione con tensori con shape diversa.\n");
        exit(EXIT_FAILURE);
    }

    size_t num_elements = tensor_get_num_elements(a->n_dim, a->shape);
    check_boolean_tensor(a, num_elements);
    check_boolean_tensor(b, num_elements);

    Tensor *result = tensor_create(a->n_dim, a->shape);

    #pragma omp parallel for
    for (size_t i = 0; i < num_elements; i++) {
        result->tensor_buffer->data[i] = (a->tensor_buffer->data[i] == 1.0f || b->tensor_buffer->data[i] == 1.0f) ? 1.0f : 0.0f;
    }

    return result;
}
Tensor* tensor_not(Tensor *a){
    if (a == NULL) return NULL;

    size_t num_elements = tensor_get_num_elements(a->n_dim, a->shape);
    check_boolean_tensor(a, num_elements);

    Tensor *result = tensor_create(a->n_dim, a->shape);

    #pragma omp parallel for
    for (size_t i = 0; i < num_elements; i++) {
        result->tensor_buffer->data[i] = (a->tensor_buffer->data[i] == 1.0f) ? 0.0f : 1.0f;
    }

    return result;
}

// Operazioni di selezione
Tensor* tensor_select(Tensor* m, Tensor* a, Tensor* b){
    if (a == NULL || b == NULL || m==NULL) return NULL;
    if (!tensor_have_same_shape(m, a) || !tensor_have_same_shape(m, b)){
        fprintf(stderr, "Errore di esecuzione: Impossibile eseguire l'operazione con tensori con shape diversa.\n");
        exit(EXIT_FAILURE);
    }

    size_t num_elements = tensor_get_num_elements(m->n_dim, m->shape);
    check_boolean_tensor(m, num_elements);

    Tensor* result = tensor_create(m->n_dim, m->shape);
    #pragma omp parallel for
    for (size_t i = 0; i < num_elements; i++) {
        result->tensor_buffer->data[i] = (m->tensor_buffer->data[i]==1.0f) ? a->tensor_buffer->data[i] : b->tensor_buffer->data[i];
    }
    
    return result;
}

// Operazioni specifiche
Tensor* tensor_matrix_prod(Tensor *a, Tensor *b){
    // Vengono distinti i vari casi a seconda delle dimensioni dei tensori
    if (a == NULL || b == NULL) return NULL;

    // Caso 1: matrice x matrice
    if (a->n_dim == 2 && b->n_dim == 2) {
        int M = a->shape[0];
        int N1 = a->shape[1];
        int N2 = b->shape[0];
        int P = b->shape[1];

        if (N1 != N2) {
            fprintf(stderr, "Errore di esecuzione: Dimensioni interne incompatibili per @ (%dx%d e %dx%d).\n", M, N1, N2, P);
            exit(EXIT_FAILURE);
        }

        int32_t res_shape[2] = {M, P};
        Tensor *result = tensor_create(2, res_shape);

        #pragma omp parallel for
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < P; j++) {
                float sum = 0.0f;
                for (int k = 0; k < N1; k++) {
                    float val_a = a->tensor_buffer->data[i * N1 + k];
                    float val_b = b->tensor_buffer->data[k * P + j];
                    sum += val_a * val_b;
                }
                result->tensor_buffer->data[i * P + j] = sum;
            }
        }
        return result;
    }

    // Caso 2: vettore colonna x vettore riga
    if (a->n_dim == 2 && b->n_dim == 1) {
        int M = a->shape[0];
        int N = a->shape[1];
        
        if (N != b->shape[0]) {
            fprintf(stderr, "Errore: Dimensioni incompatibili per 2D@1D (%dx%d e %d).\n", M, N, b->shape[0]);
            exit(EXIT_FAILURE);
        }

        int32_t res_shape[1] = {M};
        Tensor *result = tensor_create(1, res_shape);

        #pragma omp parallel for
        for (int i = 0; i < M; i++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += a->tensor_buffer->data[i * N + k] * b->tensor_buffer->data[k];
            }
            result->tensor_buffer->data[i] = sum;
        }
        return result;
    }

    // Caso 3: vettore riga x vettore colonna
    if (a->n_dim == 1 && b->n_dim == 2) {
        int N = a->shape[0];
        int P = b->shape[1];
        
        if (N != b->shape[0]) {
            fprintf(stderr, "Errore: Dimensioni incompatibili per 1D@2D (%d e %dx%d).\n", N, b->shape[0], P);
            exit(EXIT_FAILURE);
        }

        int32_t res_shape[1] = {P};
        Tensor *result = tensor_create(1, res_shape);

        #pragma omp parallel for
        for (int j = 0; j < P; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += a->tensor_buffer->data[k] * b->tensor_buffer->data[k * P + j];
            }
            result->tensor_buffer->data[j] = sum;
        }
        return result;
    }
    return NULL;
}

Tensor* tensor_dot_prod(Tensor* a, Tensor* b){
    if (a==NULL || b==NULL) return NULL;
    if (!tensor_have_same_shape(a, b)){
        fprintf(stderr, "Errore di esecuzione: Impossibile eseguire l'operazione con tensori con shape diversa.\n");
        exit(EXIT_FAILURE);
    }
    if (a->n_dim!=1){
        fprintf(stderr, "Errore di esecuzione: Operazione definita per tensori ad una dimensione\n");
        exit(EXIT_FAILURE);
    }

    int32_t res_shape[1] = {1}; // Il risultato è uno scalare (un singolo numero)
    Tensor *result = tensor_create(1, res_shape);
    
    float dot_sum = 0.0f;
    
    #pragma omp parallel for reduction(+:dot_sum)
    for (int i = 0; i < a->shape[0]; i++) {
        dot_sum += a->tensor_buffer->data[i] * b->tensor_buffer->data[i];
    }
    
    result->tensor_buffer->data[0] = dot_sum;
    return result;
}

// Operazioni sulla forma dei tensori
Tensor* tensor_reshape(Tensor* a, Tensor* s){
    if (a == NULL || s == NULL) return NULL;
    
    int num_shape_dims = tensor_get_num_elements(s->n_dim, s->shape);
    if (num_shape_dims != 2) {
        fprintf(stderr, "Errore di esecuzione: Il tensore forma deve avere 2 elementi.\n");
        exit(EXIT_FAILURE);
    }
    
    int32_t new_shape[2] = {0, 0};
    size_t new_total_elements = 1;

    new_shape[0] = (int32_t)s->tensor_buffer->data[0];
    new_shape[1] = (int32_t)s->tensor_buffer->data[1];
    new_total_elements = new_shape[0] * new_shape[1];
    
    if (tensor_get_num_elements(a->n_dim, a->shape) != new_total_elements) {
        fprintf(stderr, "Errore di esecuzione: Il numero di elementi (%zu) non coincide con la nuova forma (%zu).\n", 
                tensor_get_num_elements(a->n_dim, a->shape), new_total_elements);
        exit(EXIT_FAILURE);
    }

    Tensor* result = (Tensor*)malloc(sizeof(Tensor));
    if (result == NULL) {
        perror("Errore allocazione memoria per reshape");
        exit(EXIT_FAILURE);
    }
    
    result->n_dim = 2;
    result->shape[0] = new_shape[0];
    result->shape[1] = new_shape[1];
    
    result->tensor_buffer = a->tensor_buffer;
    result->tensor_buffer->ref_count++;
    
    return result;
}
Tensor* tensor_ravel(Tensor* t){
    if (t==NULL) return NULL;
    if (t->n_dim==1) return t;

    Tensor* result = (Tensor*)malloc(sizeof(Tensor));
    result->n_dim = 1;
    result->shape[0] = tensor_get_num_elements(2, t->shape);
    result->tensor_buffer = t->tensor_buffer;
    result->tensor_buffer->ref_count++;

    return result;
}
Tensor* tensor_shape(Tensor* t) {
    if (t == NULL) return NULL;

    int32_t res_shape[1] = { t->n_dim };

    Tensor* result = tensor_create(1, res_shape);

    for (int i = 0; i < t->n_dim; i++) {
        result->tensor_buffer->data[i] = (float)t->shape[i]; 
    }

    return result;
}

// Operazioni elemento per elemento
Tensor* tensor_relu(Tensor* t){
    if (t == NULL) return NULL;

    Tensor* result = tensor_create(t->n_dim, t->shape);
    
    size_t num_elements = tensor_get_num_elements(t->n_dim, t->shape);
    
    #pragma omp parallel for
    for (size_t i = 0; i < num_elements; i++){
        if (t->tensor_buffer->data[i] < 0.0f) {
            result->tensor_buffer->data[i] = 0.0f;
        } else {
            result->tensor_buffer->data[i] = t->tensor_buffer->data[i];
        }
    }

    return result;
}
Tensor* tensor_min(Tensor* a, Tensor* b){
    if (a == NULL || b == NULL) return NULL;
    if (!tensor_have_same_shape(a, b)){
        fprintf(stderr, "Errore di esecuzione: Impossibile eseguire l'operazione con tensori con shape diversa.\n");
        exit(EXIT_FAILURE);
    }

    Tensor* result = tensor_create(a->n_dim, a->shape);
    
    size_t num_elements = tensor_get_num_elements(a->n_dim, a->shape);
    
    #pragma omp parallel for
    for (size_t i = 0; i < num_elements; i++){
        
        if (a->tensor_buffer->data[i] < b->tensor_buffer->data[i]) {
            result->tensor_buffer->data[i] = a->tensor_buffer->data[i];
        } else {
            result->tensor_buffer->data[i] = b->tensor_buffer->data[i];        
        }
    }

    return result;
}
Tensor* tensor_max(Tensor* a, Tensor* b){
    if (a == NULL || b == NULL) return NULL;
    if (!tensor_have_same_shape(a, b)){
        fprintf(stderr, "Errore di esecuzione: Impossibile eseguire l'operazione con tensori con shape diversa.\n");
        exit(EXIT_FAILURE);
    }

    Tensor* result = tensor_create(a->n_dim, a->shape);
    
    size_t num_elements = tensor_get_num_elements(a->n_dim, a->shape);
    
    #pragma omp parallel for
    for (size_t i = 0; i < num_elements; i++){
        if (a->tensor_buffer->data[i] > b->tensor_buffer->data[i]) {
            result->tensor_buffer->data[i] = a->tensor_buffer->data[i];
        } else {
            result->tensor_buffer->data[i] = b->tensor_buffer->data[i];        
        }
    }

    return result;
}

// Operazioni di riduzione
Tensor* tensor_sum(Tensor* t){
    if (t == NULL) return NULL;

    int32_t res_shape[1] = {1};
    Tensor* result = tensor_create(1, res_shape);
    if (result == NULL) {
        fprintf(stderr, "Errore: Allocazione fallita in tensor_sum.\n");
        exit(EXIT_FAILURE);
    }

    size_t num_elements = tensor_get_num_elements(t->n_dim, t->shape);
    
    float total_sum = 0.0f;
    #pragma omp parallel for reduction(+:total_sum)
    for (size_t i = 0; i < num_elements; i++) {
        total_sum += t->tensor_buffer->data[i];
    }

    result->tensor_buffer->data[0] = total_sum;

    return result;
}

// Operazioni di filling di tensori
Tensor* tensor_fill(Tensor* s, Tensor* v){
    if (s == NULL || v == NULL) return NULL;

    if (s->n_dim != 1) {
        fprintf(stderr, "Errore di esecuzione: Il tensore forma (s) deve essere 1D.\n");
        exit(EXIT_FAILURE);
    }
    
    int len_s = s->shape[0]; 
    if (len_s < 1 || len_s > 2) {
        fprintf(stderr, "Errore di esecuzione: Il tensore forma (s) deve avere 1 o 2 elementi.\n");
        exit(EXIT_FAILURE);
    }

    size_t v_elements = tensor_get_num_elements(v->n_dim, v->shape);
    if (v_elements == 0) {
        fprintf(stderr, "Errore di esecuzione: Il tensore valori (v) è vuoto.\n");
        exit(EXIT_FAILURE);
    }

    int32_t new_n_dim = len_s;
    int32_t new_shape[2] = {0, 0};
    
    new_shape[0] = (int32_t)s->tensor_buffer->data[0];
    if (len_s == 2) {
        new_shape[1] = (int32_t)s->tensor_buffer->data[1];
    }

    if (new_shape[0] <= 0 || (len_s == 2 && new_shape[1] <= 0)) {
        fprintf(stderr, "Errore di esecuzione: Le dimensioni per il fill devono essere > 0.\n");
        exit(EXIT_FAILURE);
    }

    Tensor* result = tensor_create(new_n_dim, new_shape);
    if (result == NULL) {
        fprintf(stderr, "Errore di esecuzione: Fallita allocazione per tensor_fill.\n");
        exit(EXIT_FAILURE);
    }

    size_t res_elements = tensor_get_num_elements(new_n_dim, new_shape);

    #pragma omp parallel for
    for (size_t i = 0; i < res_elements; i++) {
        result->tensor_buffer->data[i] = v->tensor_buffer->data[i % v_elements];
    }

    return result;
}

// Operazioni di utilità
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

