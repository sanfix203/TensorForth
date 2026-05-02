// Gabriele Sanfilippo SM3201618
// File non previsto nella consegna ma utilizzato a fini di debug e test

#ifndef DEBUG_H
#define DEBUG_H

#include <stdio.h>

// flag -DDEBUG
#ifdef DEBUG
#define DEBUG_PRINT(fmt, ...) \
    fprintf(stdout, "[DEBUG] %s:%d:%s(): " fmt, __FILE__, __LINE__, __func__, ##__VA_ARGS__)
#else
    #define DEBUG_PRINT(fmt, ...) do {} while (0)
#endif

#endif