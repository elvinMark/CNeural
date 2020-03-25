#include "neural.h"

int main(){
	EMLPNetwork *nn;
	ECategoricalNetwork *cn;
	EMatrix *in,*out;
	EMatrix *out1;
	EMatrix *o;

	initMLPNetwork(&nn);
	initCategoricalNetwork(&cn);

	createMatrix(4,2,&in);
	createMatrix(4,1,&out);
	createMatrix(4,2,&out1);

	in->data[0][0] = 0;
	in->data[0][1] = 0;
	in->data[1][0] = 0;
	in->data[1][1] = 1;
	in->data[2][0] = 1;
	in->data[2][1] = 0;
	in->data[3][0] = 1;
	in->data[3][1] = 1;

	out->data[0][0] = 1;
	out->data[1][0] = 2;
	out->data[2][0] = 0;
	out->data[3][0] = 1;

	out1->data[0][0] = 1;
	out1->data[0][1] = 0;
	out1->data[1][0] = 0;
	out1->data[1][1] = 1;
	out1->data[2][0] = 0;
	out1->data[2][1] = 1;
	out1->data[3][0] = 1;
	out1->data[3][1] = 0;

	//addMLPInputLayer(nn,2);
	//addMLPLayer(nn,5,3);
	//addMLPOutputLayer(nn,2,1);
	
	addCategoricalInputLayer(cn,2);
	addCategoricalLayer(cn,5,1);
	addCategoricalLayer(cn,3,1);
	addCategoricalOutputLayer(cn);
	//addSoftMaxLayer(nn,2);

	//printMatrix(in);
	//printMatrix(out);

	//printNetwork(nn);

	//forwardNetwork(nn,in,&o);
	//printMatrix(o);

	//trainMLPNetwork(nn,in,out1,500,0.1);
	trainCategoricalNetwork(cn,in,out,500,0.01);
	//forwardMLPNetwork(nn,in,&o);
	forwardCategoricalNetwork(cn,in,&o);
	printMatrix(o);
}