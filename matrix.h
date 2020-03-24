#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
#include <math.h>

typedef struct EMatrix{
	double** data;
	int rows;
	int cols;
}EMatrix;


//Create a new matrix 
void createMatrix(int r, int c, EMatrix** m);
//Filling matrices
void onesMatrix(EMatrix *m);
void zerosMatrix(EMatrix *m);
//Print Matrix
void printMatrix(EMatrix *m);
//Basic operations between Matrices
void addMatrix(EMatrix *m1, EMatrix *m2, EMatrix **t);
void addSelfMatrix(EMatrix *m1,EMatrix *m2);
void diffMatrix(EMatrix *m1, EMatrix *m2, EMatrix **t);
void diffSelfMatrix(EMatrix *m1,EMatrix *m2);
void dotMatrix(EMatrix *m1, EMatrix *m2, EMatrix **t);
void timesMatrix(EMatrix *m1, EMatrix *m2, EMatrix **t);
void timesSelfMatrix(EMatrix *m1, EMatrix *m2);
void timesNumMatrix(double s,EMatrix *m, EMatrix **t);
void timesSelfNumMatrix(double s,EMatrix *m);
void transposeMatrix(EMatrix *m,EMatrix **t);
//Activation Functions
double actFunSigmoid(double x,bool diff);
double actFunRelu(double x,bool diff);
double actFunLinear(double x,bool diff);
double actFunTanh(double x,bool diff);
void actFunSigmoidMatrix(EMatrix *m, EMatrix **t, bool diff);
void actFunReluMatrix(EMatrix *m, EMatrix **t, bool diff);
void actFunLinearMatrix(EMatrix *m, EMatrix **t, bool diff);
void actFunTanhMatrix(EMatrix *m, EMatrix **t, bool diff);
