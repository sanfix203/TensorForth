TARGET = tensorforth

CC = gcc

CFLAGS = -Wall -Wextra -fopenmp -std=c11

LDFLAGS = -lm

SRC = main.c tensor.c stack.c operations.c

OBJ = $(SRC:.c=.o)

all: CFLAGS += -O3
all: $(TARGET)

debug: CFLAGS += -DDEBUG -g -O0
debug: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJ) $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)