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
            float temp_values[10000]; // Buffer temporaneo per i numeri
            int count = 0;
            
            // Continua a leggere finché non trovi la chiusura del tensore
            while (1) {
                ch = fgetc(file);
                if (ch == ']') break;
                if (ch == EOF) {
                    fprintf(stderr, "Errore di sintassi: File terminato inaspettatamente. Manca ']'.\n");
                    exit(EXIT_FAILURE);
                }
                if (isspace(ch)) continue;

                // Remissione del carattere nello stream per farlo leggere a fscanf
                ungetc(ch, file); 
                
                float val;
                if (fscanf(file, "%f", &val) == 1) {
                    temp_values[count++] = val;
                } else {
                    // Non è un float
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
        }
        
        // * Caso B: Inizio di una stringa 
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
        }
        
        // * Caso C: operatore
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

    // Pulizia finale
    stack_cleanup(&stack);
    fclose(file);
    return 0;
}