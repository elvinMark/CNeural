gcc -c neural.c layer.c matrix.c main.c
gcc neural.o layer.o matrix.o main.o -lm
