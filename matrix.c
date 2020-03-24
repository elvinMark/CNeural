#include "matrix.h"

//Create a new matrix 
void createMatrix(int r, int c, EMatrix** m){
	int i,j;
	//srand(time(NULL));
	*m = (EMatrix* )malloc(sizeof(EMatrix));
	(*m)->rows = r;
	(*m)->cols = c;
	(*m)->data = (double**)malloc(sizeof(double *)*r);
	for(i=0;i<r;i++){
		(*m)->data[i] = (double*)malloc(sizeof(double)*c);
		for(j=0;j<c;j++)
			(*m)->data[i][j] = (rand()%1000)/1000.0;
	}
}
//Filling matrices
void onesMatrix(EMatrix *m){
	int i,j;
	for(i=0;i<m->rows;i++){
		for(j=0;j<m->cols;j++)
			m->data[i][j] = 1;
	}
}
void zerosMatrix(EMatrix *m){
	int i,j;
	for(i=0;i<m->rows;i++){
		for(j=0;j<m->cols;j++)
			m->data[i][j] = 0;
	}
}
//Print Matrix
void printMatrix(EMatrix *m){
	int i,j;
	for(i=0;i<m->rows;i++){
		for(j=0;j<m->cols;j++)
			printf("%f ",m->data[i][j]);
		printf("\n");	
	}
	printf("\n");
}
//Basic operations between Matrices
void addMatrix(EMatrix *m1, EMatrix *m2, EMatrix **t){
	int i,j;
	createMatrix(m1->rows,m1->cols,t);
	for(i=0;i<m1->rows;i++){
		for(j=0;j<m1->cols;j++)
			(*t)->data[i][j] = m1->data[i][j] + m2->data[i][j];
	}
}
void addSelfMatrix(EMatrix *m1, EMatrix *m2){
	int i,j;
	for(i=0;i<m1->rows;i++){
		for(j=0;j<m1->cols;j++)
			m1->data[i][j] = m1->data[i][j] + m2->data[i][j];
	}
}
void diffMatrix(EMatrix *m1, EMatrix *m2, EMatrix **t){
	int i,j;
	createMatrix(m1->rows,m1->cols,t);
	for(i=0;i<m1->rows;i++){
		for(j=0;j<m1->cols;j++)
			(*t)->data[i][j] = m1->data[i][j] - m2->data[i][j];
	}
}
void diffSelfMatrix(EMatrix *m1, EMatrix *m2){
	int i,j;
	for(i=0;i<m1->rows;i++){
		for(j=0;j<m1->cols;j++)
			m1->data[i][j] = m1->data[i][j] - m2->data[i][j];
	}
}
void dotMatrix(EMatrix *m1, EMatrix *m2, EMatrix **t){
	int i,j,k;
	double s;
	createMatrix(m1->rows,m2->cols,t);
	for(i=0;i<m1->rows;i++){
		for(j=0;j<m2->cols;j++){
			s = 0;
			for(k=0;k<m1->cols;k++)
				s += m1->data[i][k]*m2->data[k][j];
			(*t)->data[i][j] = s;
		}
	}
}
void timesMatrix(EMatrix *m1, EMatrix *m2, EMatrix **t){
	int i,j;
	createMatrix(m1->rows,m1->cols,t);
	for(i=0;i<m1->rows;i++){
		for(j=0;j<m1->cols;j++)
			(*t)->data[i][j] = m1->data[i][j] * m2->data[i][j];
	}	
}
void timesSelfMatrix(EMatrix *m1, EMatrix *m2){
	int i,j;
	for(i=0;i<m1->rows;i++){
		for(j=0;j<m1->cols;j++)
			m1->data[i][j] = m1->data[i][j] * m2->data[i][j];
	}	
}
void timesNumMatrix(double s,EMatrix *m, EMatrix **t){
	int i,j;
	createMatrix(m->rows,m->cols,t);
	for(i=0;i<m->rows;i++){
		for(j=0;j<m->cols;j++)
			(*t)->data[i][j] = s * m->data[i][j];
	}	
}
void timesSelfNumMatrix(double s,EMatrix *m){
	int i,j;
	for(i=0;i<m->rows;i++){
		for(j=0;j<m->cols;j++)
			m->data[i][j] = s * m->data[i][j];
	}	
}
void transposeMatrix(EMatrix *m,EMatrix **t){
	int i,j;
	createMatrix(m->cols,m->rows,t);
	for(i=0;i<m->rows;i++)
		for(j=0;j<m->cols;j++)
			(*t)->data[j][i] = m->data[i][j];
}

double actFunSigmoid(double x,bool diff){
	if(diff)
		return x*(1.0-x);
	return 1.0/(1 + exp(-x));
}
double actFunRelu(double x,bool diff){
	if(diff)
		return x>=0?1:0.00001;
	return x>=0?x:0.00001*x;
}
double actFunLinear(double x,bool diff){
	if(diff)
		return 1;
	return x;
}
double actFunTanh(double x,bool diff){
	if(diff)
		return (1-x*x)/2;
	return (1 - exp(-x))/(1 + exp(-x));
}
void actFunSigmoidMatrix(EMatrix *m, EMatrix **t, bool diff){
	int i,j;
	createMatrix(m->rows,m->cols,t);
	for(i=0;i<m->rows;i++){
		for(j =0;j<m->cols;j++)
			(*t)->data[i][j] = actFunSigmoid(m->data[i][j],diff);
	}
}
void actFunSelfSigmoidMatrix(EMatrix *m, bool diff){
	int i,j;
	for(i=0;i<m->rows;i++){
		for(j =0;j<m->cols;j++)
			m->data[i][j] = actFunSigmoid(m->data[i][j],diff);
	}
}
void actFunReluMatrix(EMatrix *m, EMatrix **t, bool diff){
	int i,j;
	createMatrix(m->rows,m->cols,t);
	for(i=0;i<m->rows;i++){
		for(j =0;j<m->cols;j++)
			(*t)->data[i][j] = actFunRelu(m->data[i][j],diff);
	}
}
void actFunSelfReluMatrix(EMatrix *m, bool diff){
	int i,j;
	for(i=0;i<m->rows;i++){
		for(j =0;j<m->cols;j++)
			m->data[i][j] = actFunRelu(m->data[i][j],diff);
	}
}
void actFunLinearMatrix(EMatrix *m, EMatrix **t, bool diff){
	int i,j;
	createMatrix(m->rows,m->cols,t);
	for(i=0;i<m->rows;i++){
		for(j =0;j<m->cols;j++)
			(*t)->data[i][j] = actFunLinear(m->data[i][j],diff);
	}
}
void actFunSelfLinearMatrix(EMatrix *m, bool diff){
	int i,j;
	for(i=0;i<m->rows;i++){
		for(j =0;j<m->cols;j++)
			m->data[i][j] = actFunLinear(m->data[i][j],diff);
	}
}
void actFunTanhMatrix(EMatrix *m, EMatrix **t, bool diff){
	int i,j;
	createMatrix(m->rows,m->cols,t);
	for(i=0;i<m->rows;i++){
		for(j =0;j<m->cols;j++)
			(*t)->data[i][j] = actFunTanh(m->data[i][j],diff);
	}
}
void actFunSelfTanhMatrix(EMatrix *m, bool diff){
	int i,j;
	for(i=0;i<m->rows;i++){
		for(j =0;j<m->cols;j++)
			m->data[i][j] = actFunTanh(m->data[i][j],diff);
	}
}
