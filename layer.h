#include "matrix.h"

typedef struct ELayer{
	EMatrix *weights;
	EMatrix *bias;
	int n_inputs;
	int n_outputs;
	EMatrix *inData;
	EMatrix *outData;
	EMatrix *delta;
	int type;
}ELayer;

extern int TYPE_SIGMOID;
extern int TYPE_RELU;
extern int TYPE_TANH;
extern int TYPE_LINEAR;

//Create new Layer
void createLayer(int ni,int no,int type,ELayer **l);
void createSoftMaxLayer(int ni,ELayer **l);
//Print Layer
void printLayer(ELayer *l);
//Functions 
void forwardLayer(ELayer *l, EMatrix *in, EMatrix **out);
void backwardLayer(ELayer *l, EMatrix *err, EMatrix **propErr);
void updateLayer(ELayer *l,double learning_rate);