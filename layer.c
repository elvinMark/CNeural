#include "layer.h"

int TYPE_SIGMOID = 0;
int TYPE_RELU = 1;
int TYPE_LINEAR = 2;
int TYPE_TANH = 3;

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
	if(l->weights == NULL){
		int i,j;
		double s;
		l->inData = in;
		createMatrix(in->rows,l->n_outputs,&(l->outData));
		for(i=0;i<in->rows;i++){
			s = 0;
			for(j=0;j<in->cols;j++)
				s += in->data[i][j];
			for(j=0;j<in->cols;j++)
				l->outData->data[i][j] = in->data[i][j]/s;
		}
		*out = l->outData;
		return;
	}
	EMatrix *s,*tmp,*b;
	l->inData = in;
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
	if(l->weights == NULL){
		EMatrix *t;
		actFunSigmoidMatrix(l->outData,&t,true);
		timesMatrix(err,t,&(l->delta));
		*propErr = l->delta; 
		return;
	}
	EMatrix *s,*t;
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
	if(l->weights == NULL){
		return;
	}
	EMatrix *s,*t;
	EMatrix *ones;
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
