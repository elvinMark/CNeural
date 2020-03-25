#include "layer.h"

int TYPE_SIGMOID = 0;
int TYPE_RELU = 1;
int TYPE_LINEAR = 2;
int TYPE_TANH = 3;
int TYPE_SOFTMAX = 4;

//Create new Layer
void createLayer(int ni,int no,int type,ELayer **l){
	(*l) = (ELayer*) malloc(sizeof(ELayer));
	createMatrix(ni,no,&((*l)->weights));
	createMatrix(1,no,&((*l)->bias));
	(*l)->n_inputs = ni;
	(*l)->n_outputs = no;
	(*l)->type = type;
}
void createSoftMaxLayer(int ni,ELayer **l){
	(*l) = (ELayer*) malloc(sizeof(ELayer));
	(*l)->n_inputs = ni;
	(*l)->n_outputs = ni;
	(*l)->type = TYPE_SOFTMAX;
}
//Print Layer
void printLayer(ELayer *l){
	printf("Inputs: %d \t Outputs: %d\n",l->n_inputs,l->n_outputs);
	printf("Weights:\n");
	printMatrix(l->weights);
	printf("Bias:\n");
	printMatrix(l->bias);
}
//Functions 
void forwardLayer(ELayer *l, EMatrix *in, EMatrix **out){
	EMatrix *s,*tmp,*b;
	l->inData = in;
	if(l->type == TYPE_SOFTMAX){
		actFunSoftMaxMatrix(in,&(l->outData));
		*out = l->outData;
		return;
	}
	dotMatrix(in,l->weights,&s);
	createMatrix(in->rows,1,&tmp);
	onesMatrix(tmp);
	dotMatrix(tmp,l->bias,&b);
	addMatrix(s,b,&tmp);
	switch(l->type){
		case 0:
		actFunSigmoidMatrix(tmp,&(l->outData),false);
		break;
		case 1:
		actFunReluMatrix(tmp,&(l->outData),false);
		break;
		case 2:
		actFunLinearMatrix(tmp,&(l->outData),false);
		break;
		case 3:
		actFunTanhMatrix(tmp,&(l->outData),false);
		break;
		default:
		break;
	}
	*out = l->outData;
}
void backwardLayer(ELayer *l, EMatrix *err, EMatrix **propErr){
	EMatrix *s,*t;
	int i,j;
	s = err;
	if(l->type == TYPE_SOFTMAX){
		createMatrix(err->rows,l->n_outputs,propErr);
		for(i=0;i<err->rows;i++){
			for(j=0;j<l->n_outputs;j++){
				if(s->data[i][0] == j)
					(*propErr)->data[i][j] = (l->outData->data[i][j] - 1);
				else
					(*propErr)->data[i][j] = l->outData->data[i][j];
			}
		}
		return;
	}
	switch(l->type){
		case 0:
		actFunSigmoidMatrix(l->outData,&s,true);
		break;
		case 1:
		actFunReluMatrix(l->outData,&s,true);
		break;
		case 2:
		actFunLinearMatrix(l->outData,&s,true);
		break;
		case 3:
		actFunTanhMatrix(l->outData,&s,true);
		break;
		default:
		break;
	}
	timesMatrix(err,s,&(l->delta));
	transposeMatrix(l->weights,&t);
	dotMatrix(l->delta,t,propErr);
}
void updateLayer(ELayer *l,double learning_rate){
	EMatrix *s,*t;
	EMatrix *ones;
	if(l->type == TYPE_SOFTMAX)
		return;
	transposeMatrix(l->inData,&t);
	createMatrix(1,l->inData->rows,&ones);
	onesMatrix(ones);
	dotMatrix(t,l->delta,&s);
	dotMatrix(ones,l->delta,&t);
	timesSelfNumMatrix(learning_rate,s);
	timesSelfNumMatrix(learning_rate,t);
	diffSelfMatrix(l->weights,s);
	diffSelfMatrix(l->bias,t);
}
