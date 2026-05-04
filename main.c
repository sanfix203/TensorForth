// Gabriele Sanfilippo SM3201618

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "tensor.h"
#include "stack.h"
#include "debug.h"
#include "operations.h"

// Dimensione massima per il buffer di lettura temporaneo
#define MAX_BUFFER_SIZE 2048

int main(int argc, char *argv[]) {
    // Controllo degli argomenti 
    if (argc != 2) {
        fprintf(stderr, "Uso: %s [nome file sorgente]\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    // Apertura del file sorgente
    FILE *file = fopen(argv[1], "r");
    if (file == NULL) {
        perror("Errore nell'apertura del file sorgente");
        exit(EXIT_FAILURE);
    }

    // Inizializzazione dello Stack
    Stack stack;
    stack_init(&stack);

    int ch;
    while ((ch = fgetc(file)) != EOF) {
        // Ignora spazi
        if (isspace(ch)) {
            continue;
        }

        // * Caso A: inizio tensore
        if (ch == '[') {
            // Controllo 1: spazio obbligatorio DOPO '['
            int next_ch = fgetc(file);
            if (!isspace(next_ch)) {
                fprintf(stderr, "Errore di sintassi: Spazio obbligatorio\n");
                exit(EXIT_FAILURE);
            }
            ungetc(next_ch, file); 

            float temp_values[10000];
            int count = 0;
            
            int last_was_space = 1; 
            
            // Continua a leggere finché non trovi la chiusura del tensore
            while (1) {
                ch = fgetc(file);
                
                if (ch == ']') {
                    // Controllo 2: spazio obbligatorio PRIMA di ']'
                    if (!last_was_space) {
                        fprintf(stderr, "Errore di sintassi: Spazio obbligatorio\n");
                        exit(EXIT_FAILURE);
                    }
                    break;
                }
                
                if (ch == EOF) {
                    fprintf(stderr, "Errore di sintassi: File terminato inaspettatamente.\n");
                    exit(EXIT_FAILURE);
                }
                
                if (isspace(ch)) {
                    last_was_space = 1;
                    continue;
                }

                last_was_space = 0; 
                ungetc(ch, file); 
                
                float val;
                if (fscanf(file, "%f", &val) == 1) {
                    temp_values[count++] = val;
                } else {
                    fprintf(stderr, "Errore di sintassi: valore non numerico nel tensore.\n");
                    exit(EXIT_FAILURE);
                }
            }
            
            int32_t shape[MAX_DIM] = {count, 0};
            Tensor *t = tensor_create(1, shape);
            for(int i = 0; i < count; i++) {
                t->tensor_buffer->data[i] = temp_values[i];
            }
            stack_push_tensor(&stack, t);
            DEBUG_PRINT("Letto tensore con %d elementi.\n", count);
        } // * Caso B: Inizio di una stringa 
        else if (ch == '"') {
            char filename[MAX_BUFFER_SIZE];
            int i = 0;
            
            while (1) {
                ch = fgetc(file);
                if (ch == '"') {
                    break; 
                }
                if (ch == EOF) {
                    fprintf(stderr, "Errore di sintassi: File terminato inaspettatamente (manca '\"').\n");
                    exit(EXIT_FAILURE);
                }
            
                if (i < MAX_BUFFER_SIZE - 1) {
                    filename[i++] = (char)ch;
                }
            }
            filename[i] = '\0'; // Terminatore di stringa
            
            stack_push_string(&stack, filename);
            DEBUG_PRINT("Inserito filename nello stack: '%s'\n", filename);
        } // * Caso C: operatore
        else {
            char op = (char)ch;
            
            switch(op) {
                case '+':{
                    DEBUG_PRINT("operatore '%c'\n", op);
                    Tensor *b = stack_pop_tensor(&stack);
                    Tensor *a = stack_pop_tensor(&stack);
                    Tensor *result = tensor_add(a, b);

                    stack_push_tensor(&stack, result);
                    break;}
                case '-':{
                    DEBUG_PRINT("operatore '%c'\n", op);
                    Tensor *b = stack_pop_tensor(&stack);
                    Tensor *a = stack_pop_tensor(&stack);
                    Tensor *result = tensor_sub(a, b);

                    stack_push_tensor(&stack, result);
                    break;}
                case '*':{
                    DEBUG_PRINT("operatore '%c'\n", op);
                    Tensor *b = stack_pop_tensor(&stack);
                    Tensor *a = stack_pop_tensor(&stack);
                    Tensor *result = tensor_mul(a, b);

                    stack_push_tensor(&stack, result);
                    break;}
                case '>':{
                    DEBUG_PRINT("operatore '%c'\n", op);
                    Tensor *a = stack_pop_tensor(&stack);
                    Tensor *b = stack_pop_tensor(&stack);
                    Tensor *result = tensor_greater(a, b);

                    stack_push_tensor(&stack, result);
                    break;}
                case '=':{
                    DEBUG_PRINT("operatore '%c'\n", op);
                    Tensor *b = stack_pop_tensor(&stack);
                    Tensor *a = stack_pop_tensor(&stack);
                    Tensor *result = tensor_equal(a, b);

                    stack_push_tensor(&stack, result);
                    break;}
                case '<':{
                    DEBUG_PRINT("operatore '%c'\n", op);
                    Tensor *a = stack_pop_tensor(&stack);
                    Tensor *b = stack_pop_tensor(&stack);
                    Tensor *result = tensor_less(a, b);

                    stack_push_tensor(&stack, result);
                    break;}
                case '&': {
                    Tensor *b = stack_pop_tensor(&stack);
                    Tensor *a = stack_pop_tensor(&stack);
                    Tensor *result = tensor_and(a, b);

                    stack_push_tensor(&stack, result);
                    break;}
                case '|': {
                    Tensor *b = stack_pop_tensor(&stack);
                    Tensor *a = stack_pop_tensor(&stack);
                    Tensor *result = tensor_or(a, b);

                    stack_push_tensor(&stack, result);
                    break;}
                case '!': {
                    Tensor *a = stack_pop_tensor(&stack);
                    Tensor *result = tensor_not(a);

                    stack_push_tensor(&stack, result);
                    break;}
                case '$': {
                    Tensor *m = stack_pop_tensor(&stack);
                    Tensor *a = stack_pop_tensor(&stack);
                    Tensor *b = stack_pop_tensor(&stack);
                    Tensor *result = tensor_select(m, a, b);

                    stack_push_tensor(&stack, result);
                    break;}
                case 'r': {
                    Tensor* s = stack_pop_tensor(&stack);
                    Tensor* a = stack_pop_tensor(&stack);
                    Tensor* result = tensor_reshape(a, s);

                    stack_push_tensor(&stack, result);
                    break;}
                case '_': {
                    Tensor *t = stack_pop_tensor(&stack);
                    Tensor *result = tensor_ravel(t);

                    stack_push_tensor(&stack, result);
                    break;}
                case '#': {
                    Tensor *t = stack_pop_tensor(&stack);
                    Tensor *result = tensor_shape(t);

                    stack_push_tensor(&stack, result);
                    break;}
                case 'R': {
                    Tensor *t = stack_pop_tensor(&stack);
                    Tensor *result = tensor_relu(t);

                    stack_push_tensor(&stack, result);
                    break;}
                case 'm': {
                    Tensor *a = stack_pop_tensor(&stack);
                    Tensor *b = stack_pop_tensor(&stack);
                    Tensor *result = tensor_min(a, b);

                    stack_push_tensor(&stack, result);
                    break;}
                case 'M': {
                    Tensor *a = stack_pop_tensor(&stack);
                    Tensor *b = stack_pop_tensor(&stack);
                    Tensor *result = tensor_max(a, b);

                    stack_push_tensor(&stack, result);
                    break;}
                case 'S': {
                    Tensor *t = stack_pop_tensor(&stack);
                    Tensor *result = tensor_sum(t);

                    stack_push_tensor(&stack, result);
                    break;}
                case '@':{
                    Tensor* a = stack_pop_tensor(&stack);
                    Tensor* b = stack_pop_tensor(&stack);
                    Tensor* result = tensor_matrix_prod(a, b);

                    stack_push_tensor(&stack, result);
                    break;}
                case '.':{
                    Tensor* a = stack_pop_tensor(&stack);
                    Tensor* b = stack_pop_tensor(&stack);
                    Tensor* result = tensor_dot_prod(a, b);

                    stack_push_tensor(&stack, result);
                    break;}
                case 'f': {
                    Tensor *v = stack_pop_tensor(&stack); // L'ultimo in cima: valori
                    Tensor *s = stack_pop_tensor(&stack); // Il penultimo: forma
                    
                    Tensor *result = tensor_fill(s, v);
                    stack_push_tensor(&stack, result);

                    break;}
                case 'd':
                    DEBUG_PRINT("operatore '%c' (dup)\n", op);
                    // stack_push(&stack, tensor_reference(stack_peek(&stack)));
                    break;
                case 'p':{
                    DEBUG_PRINT("operatore '%c' (print)\n", op);
                    Tensor *t = stack_pop_tensor(&stack);
                    tensor_print(t);
                    break;}
  
                // ecc
                
                default:
                    fprintf(stderr, "Errore: Operatore o carattere sconosciuto '%c'\n", op);
                    exit(EXIT_FAILURE);
            }
        }
    }

    stack_cleanup(&stack);
    fclose(file);
    return 0;
}