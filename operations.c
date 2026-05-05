// Gabriele Sanfilippo SM3201618

#include "operations.h"
#include "tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h> // per operazioni di comparazione con float
#include <string.h>
#include <ctype.h>


// Funzioni di supporto
static void check_boolean_tensor(Tensor *t, size_t num_elements) {
    if (t == NULL) return;
    
    for (size_t i = 0; i < num_elements; i++) {
        if (t->tensor_buffer == NULL || t->tensor_buffer->data == NULL) {
            fprintf(stderr, "Errore: Il tensore non ha dati\n");
            exit(EXIT_FAILURE);
        }
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
    if (a == NULL || b == NULL) return NULL;

    if (a->n_dim != 2 || b->n_dim != 2){
        fprintf(stderr, "Errore di esecuzione: Impossibile eseguire l'operazione con tensori 1D.\n");
        exit(EXIT_FAILURE);
    }

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
Tensor* tensor_conv2d(Tensor* a, Tensor* k) {
    if (a == NULL || k == NULL) return NULL;

    if (a->n_dim != 2 || k->n_dim != 2) {
        fprintf(stderr, "Errore di esecuzione: L'operatore di convoluzione richiede due tensori 2D.\n");
        exit(EXIT_FAILURE);
    }

    int M = a->shape[0];  // Righe immagine
    int N = a->shape[1];  // Colonne immagine
    int KM = k->shape[0]; // Righe kernel
    int KN = k->shape[1]; // Colonne kernel

    int32_t res_shape[2] = {M, N};
    Tensor* result = tensor_create(2, res_shape);
    if (result == NULL) {
        fprintf(stderr, "Errore: Allocazione fallita in tensor_conv2d.\n");
        exit(EXIT_FAILURE);
    }

    // Offset
    int pad_y = KM / 2;
    int pad_x = KN / 2;

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            
            for (int ki = 0; ki < KM; ki++) {
                for (int kj = 0; kj < KN; kj++) {
                    int r = i + ki - pad_y;
                    int c = j + kj - pad_x;

                    // Padding
                    if (r >= 0 && r < M && c >= 0 && c < N) {
                        float val_a = a->tensor_buffer->data[r * N + c];
                        float val_k = k->tensor_buffer->data[ki * KN + kj];
                        sum += val_a * val_k;
                    }
                }
            }

            result->tensor_buffer->data[i * N + j] = sum;
        }
    }

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

// Operazioni di generazione di numeri casuali
Tensor* tensor_random(Tensor* s){
    if (s == NULL) return NULL;

    if (s->n_dim != 1) {
        fprintf(stderr, "Errore di esecuzione: Il tensore forma deve essere 1D.\n");
        exit(EXIT_FAILURE);
    }

    int32_t new_n_dim = s->shape[0];
    int32_t new_shape[new_n_dim]; // Variable Length Array    

    for (int i = 0; i < new_n_dim; i++) {
        new_shape[i] = (int32_t)s->tensor_buffer->data[i];
        if (new_shape[i] <= 0) {
            fprintf(stderr, "Errore di esecuzione: Le dimensioni per '?' devono essere > 0.\n");
            exit(EXIT_FAILURE);
        }
    }

    Tensor* result = tensor_create(new_n_dim, new_shape);
    if (result == NULL) {
        fprintf(stderr, "Errore: Allocazione fallita per tensor_random.\n");
        exit(EXIT_FAILURE);
    }

    size_t num_elements = tensor_get_num_elements(new_n_dim, new_shape);

    for (size_t i = 0; i < num_elements; i++) {
        result->tensor_buffer->data[i] = (float)(rand() % 2);
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

// Operazioni di manipolazione dello stack
Tensor* tensor_duplicate(Tensor* a){
    if (a==NULL) return NULL;
    Tensor* result = (Tensor*)malloc(sizeof(Tensor));
    if (result == NULL) {
        fprintf(stderr, "Errore: Malloc fallita per il clone in tensor_duplicate.\n");
        exit(EXIT_FAILURE);
    }

    result->n_dim = a->n_dim;
    for (int i = 0; i < a->n_dim; i++) {
        result->shape[i] = a->shape[i];
    }

    result->tensor_buffer = a->tensor_buffer;
    result->tensor_buffer->ref_count++;

    return result;
}

// Lettura/scrittura da file pgm
void tensor_save_pgm(Tensor* t, const char* filename) {
    if (t == NULL || filename == NULL) return;

    if (t->n_dim != 2) {
        fprintf(stderr, "Errore: Solo i tensori 2D possono essere salvati come immagine PGM.\n");
        exit(EXIT_FAILURE);
    }

    FILE *f = fopen(filename, "wb");
    if (!f) {
        fprintf(stderr, "Errore: Impossibile creare il file '%s'\n", filename);
        exit(EXIT_FAILURE);
    }

    int height = t->shape[0];
    int width = t->shape[1];

    // Header P5 (Formato, Larghezza, Altezza, MaxVal)
    fprintf(f, "P5\n%d %d\n255\n", width, height);

    size_t num_pixels = width * height;
    unsigned char *raw_pixels = (unsigned char*)malloc(num_pixels);
    if (raw_pixels == NULL) {
        fprintf(stderr, "Errore: RAM insufficiente per il salvataggio dell'immagine\n");
        fclose(f);
        exit(EXIT_FAILURE);
    }

    // Clamping e rimappatura [0, 1] -> [0, 255]
    #pragma omp parallel for
    for (size_t i = 0; i < num_pixels; i++) {
        float val = t->tensor_buffer->data[i];
        
        // 1. Clamping inferiore e superiore
        if (val < 0.0f) val = 0.0f;
        if (val > 1.0f) val = 1.0f;
        
        // 2. Rimappatura a 255
        raw_pixels[i] = (unsigned char)(val * 255.0f);
    }

    fwrite(raw_pixels, 1, num_pixels, f);

    free(raw_pixels);
    fclose(f);
}
Tensor* tensor_load_pgm(const char* filename) {
    FILE *f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Errore: Impossibile aprire il file '%s'\n", filename);
        exit(EXIT_FAILURE);
    }

    char magic[3];
    // Magic Number (deve essere P5)
    if (fscanf(f, "%2s", magic) != 1 || strcmp(magic, "P5") != 0) {
        fprintf(stderr, "Errore: Il file '%s' non e' un PGM binario (P5)\n", filename);
        fclose(f);
        exit(EXIT_FAILURE);
    }

    int width, height, maxval;

    skip_pgm_comments(f);
    if (fscanf(f, "%d", &width) != 1) exit(EXIT_FAILURE);
    
    skip_pgm_comments(f);
    if (fscanf(f, "%d", &height) != 1) exit(EXIT_FAILURE);
    
    skip_pgm_comments(f);
    if (fscanf(f, "%d", &maxval) != 1 || maxval > 255) {
        fprintf(stderr, "Errore: Formato PGM non supportato o corrotto in '%s'\n", filename);
        fclose(f);
        exit(EXIT_FAILURE);
    }

    // Il carattere immediatamente successivo al maxval è un singolo spazio bianco da scartare
    fgetc(f);

    int32_t shape[2] = {height, width};
    Tensor* result = tensor_create(2, shape);
    if (result == NULL) {
        fprintf(stderr, "Errore: Allocazione tensore fallita per '%s'\n", filename);
        fclose(f);
        exit(EXIT_FAILURE);
    }

    size_t num_pixels = width * height;
    unsigned char *raw_pixels = (unsigned char*)malloc(num_pixels);
    if (raw_pixels == NULL) {
        fprintf(stderr, "Errore: RAM insufficiente per leggere l'immagine\n");
        exit(EXIT_FAILURE);
    }

    if (fread(raw_pixels, 1, num_pixels, f) != num_pixels) {
        fprintf(stderr, "Errore: Lettura pixel incompleta dal file '%s'\n", filename);
        exit(EXIT_FAILURE);
    }
    fclose(f);

    #pragma omp parallel for
    for (size_t i = 0; i < num_pixels; i++) {
        result->tensor_buffer->data[i] = (float)raw_pixels[i] / 255.0f;
    }

    free(raw_pixels);
    return result;
}
void skip_pgm_comments(FILE *f) {
    int ch;
    char line[256];
    while ((ch = fgetc(f)) != EOF && isspace(ch));
    while (ch == '#') {
        if (fgets(line, sizeof(line), f) == NULL) break;
        while ((ch = fgetc(f)) != EOF && isspace(ch));
    }
    if (ch != EOF) {
        ungetc(ch, f);
    }
}

// Lettura/scrittura da file binario
void tensor_save_raw(Tensor* t, const char* filename) {
    if (t == NULL || filename == NULL) return;

    FILE *f = fopen(filename, "wb");
    if (!f) {
        fprintf(stderr, "Errore: Impossibile creare il file '%s'\n", filename);
        exit(EXIT_FAILURE);
    }

    on_disk_tensor header;
    memset(&header, 0, sizeof(header)); // Azzera per sicurezza
    header.ndim = t->n_dim;
    for (int i = 0; i < t->n_dim && i < MAX_DIM; i++) {
        header.shape[i] = t->shape[i];
    }
    header.data_offset = 64; // Allineamento imposto a 64 bytes

    fwrite(&header, sizeof(header), 1, f);

    // Padding
    size_t header_size = sizeof(header);
    if (header_size < 64) {
        char padding[64] = {0};
        fwrite(padding, 1, 64 - header_size, f);
    }

    // Scrittura
    size_t num_elements = tensor_get_num_elements(t->n_dim, t->shape);
    fwrite(t->tensor_buffer->data, sizeof(float), num_elements, f);

    fclose(f);
}
Tensor* tensor_load_raw(const char* filename) {
    int fd = open(filename, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "Errore: Impossibile aprire il file '%s'\n", filename);
        exit(EXIT_FAILURE);
    }

    // Grandezza file
    struct stat st;
    if (fstat(fd, &st) < 0) {
        fprintf(stderr, "Errore di lettura stat su '%s'\n", filename);
        close(fd);
        exit(EXIT_FAILURE);
    }

    // Mappatura dell'intero file
    void *mapped_memory = mmap(NULL, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (mapped_memory == MAP_FAILED) {
        fprintf(stderr, "Errore: mmap fallita per '%s'\n", filename);
        close(fd);
        exit(EXIT_FAILURE);
    }
    close(fd);

    on_disk_tensor *header = (on_disk_tensor *)mapped_memory;

    Tensor* result = (Tensor*)malloc(sizeof(Tensor));
    result->n_dim = header->ndim;
    memcpy(result->shape, header->shape, result->n_dim * sizeof(int32_t));

    result->tensor_buffer = (TensorBuffer*)malloc(sizeof(TensorBuffer));
    result->tensor_buffer->ref_count = 1;
    result->tensor_buffer->mmap_ptr = mapped_memory; // Memorizziamo per il futuro munmap
    result->tensor_buffer->mmap_size = st.st_size;
    
    result->tensor_buffer->data = (float*)((char*)mapped_memory + header->data_offset);

    return result;
}