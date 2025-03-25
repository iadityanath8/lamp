# Compiler and Flags
CC = gcc
CFLAGS = -O3 -march=native -fopenmp -mavx -mfma
LDFLAGS =  -lcublas -lcudart -lm#-lopenblas //-I/usr/include/openblas -L/usr/lib/x86_64-linux-gnu


SRC = main.c ./src/ndarray.c ./src/value.c ./src/tensor.c
OBJ = $(SRC:.c=.o)
EXEC = main


all: $(EXEC)


$(EXEC): $(SRC)
	$(CC) $(CFLAGS) -o $(EXEC) $(SRC) $(LDFLAGS)


test: test.c ./src/ndarray.c
	$(CC) $(CFLAGS) -o test test.c ./src/ndarray.c $(LDFLAGS)


clean:
	rm -rf $(EXEC) test *.o ./src/*.o
