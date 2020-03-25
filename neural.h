#include "layer.h"

typedef struct node{
	ELayer* l;
	struct node* next;
	struct node* before;
}node;

//Multilayer Perceptron Network using Root Mean Square Loss
typedef struct EMLPNetwork{
	node* head;
	node* last;
	int n_prev;	
}EMLPNetwork;

//Categorical Network using Cross Entropy Loss
typedef struct ECategoricalNetwork{
	node* head;
	node* last;
	int n_prev;
}ECategoricalNetwork;

//Initialize MultiLayer Perceptron Nerual Network
void initMLPNetwork(EMLPNetwork **net);
//Structure the Network
void addMLPInputLayer(EMLPNetwork *net,int nin);
void addMLPLayer(EMLPNetwork *net,int nh,int type);
void addMLPOutputLayer(EMLPNetwork *net,int nout,int type);
//Functions
void forwardMLPNetwork(EMLPNetwork *net,EMatrix* in,EMatrix **t);
void backwardMLPNetwork(EMLPNetwork *net,EMatrix *err);
void updateMLPNetwork(EMLPNetwork *net,double learning_rate);
void trainMLPNetwork(EMLPNetwork *net, EMatrix *in, EMatrix *out,int maxIt,double learning_rate);


//Initialize Categorical Nerual Network
void initCategoricalNetwork(ECategoricalNetwork **net);
//Structure the Network
void addCategoricalInputLayer(ECategoricalNetwork *net,int nin);
void addCategoricalLayer(ECategoricalNetwork *net,int nh,int type);
void addCategoricalOutputLayer(ECategoricalNetwork *net);
//Functions
void forwardCategoricalNetwork(ECategoricalNetwork *net,EMatrix* in,EMatrix **t);
void backwardCategoricalNetwork(ECategoricalNetwork *net,EMatrix *err);
void updateCategoricalNetwork(ECategoricalNetwork *net,double learning_rate);
void trainCategoricalNetwork(ECategoricalNetwork *net, EMatrix *in, EMatrix *out,int maxIt,double learning_rate);