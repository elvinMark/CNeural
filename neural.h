#include "layer.h"

typedef struct node{
	ELayer* l;
	struct node* next;
	struct node* before;
}node;

typedef struct ENeuralNetwork{
	node* head;
	node* last;
	int loss_type;
}ENeuralNetwork;

extern int LOSS_RMS;
extern int LOSS_CROSS_ENTROPY;

//Init network
void initNetwork(ENeuralNetwork **net,int loss_type);
//Print Network
void printNetwork(ENeuralNetwork *net);
//Add new Layer
void addLayer(ENeuralNetwork* net,int nin,int nout,int type);
void addSoftMaxLayer(ENeuralNetwork *net,int nin);
//Functions
void forwardNetwork(ENeuralNetwork *net,EMatrix* in,EMatrix **t);
void backwardNetwork(ENeuralNetwork *net,EMatrix *err);
void updateNetwork(ENeuralNetwork *net,double learning_rate);
void calculateLossGrad(ENeuralNetwork *net, EMatrix *o, EMatrix *t, EMatrix **g);
void trainNetwork(ENeuralNetwork *net, EMatrix *in, EMatrix *out,int maxIt,double learning_rate);