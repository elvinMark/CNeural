#include "neural.h"

int main(){
	ENeuralNetwork *nn;
	EMatrix *in,*out;
	EMatrix *o;

	initNetwork(&nn,1);

	createMatrix(4,2,&in);
	createMatrix(1,4,&out);

	in->data[0][0] = 0;
	in->data[0][1] = 0;
	in->data[1][0] = 0;
	in->data[1][1] = 1;
	in->data[2][0] = 1;
	in->data[2][1] = 0;
	in->data[3][0] = 1;
	in->data[3][1] = 1;

	out->data[0][0] = 1;
	out->data[0][1] = 0;
	out->data[0][2] = 0;
	out->data[0][3] = 1;

	addLayer(nn,2,5,1);
	addLayer(nn,5,2,1);
	addSoftMaxLayer(nn,2);

	//printMatrix(in);
	//printMatrix(out);

	//printNetwork(nn);

	//forwardNetwork(nn,in,&o);
	//printMatrix(o);

	trainNetwork(nn,in,out,1000,1);
	forwardNetwork(nn,in,&o);
	printMatrix(o);
}