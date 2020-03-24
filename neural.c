#include "neural.h"

int LOSS_RMS = 0;
int LOSS_CROSS_ENTROPY = 1;

//Init Network
void initNetwork(ENeuralNetwork **net,int loss_type){
	*net = (ENeuralNetwork*) malloc(sizeof(ENeuralNetwork));
	(*net)->loss_type = loss_type;
}
//Print Network
void printNetwork(ENeuralNetwork *net){
	node *tmp;
	int count=1;
	for(tmp = net->head;tmp!=NULL;tmp = tmp->next){
		printf("Layer %d:\n",count++);
		printLayer(tmp->l);
	}
}
//Add new Layer
void addLayer(ENeuralNetwork* net,int nin,int nout,int type){
	if(net->head == NULL){
		net->head = (node*) malloc(sizeof(node));
		net->last = net->head;
		createLayer(nin,nout,type,&(net->head->l));
	}
	else{
		net->last->next = (node*) malloc(sizeof(node));
		net->last->next->before = net->last;
		createLayer(nin,nout,type,&(net->last->next->l));
		net->last = net->last->next;
	}
}
void addSoftMaxLayer(ENeuralNetwork *net,int nin){
	if(net->head == NULL){
		net->head = (node*) malloc(sizeof(node));
		net->last = net->head;
		createSoftMaxLayer(nin,&(net->head->l));
	}
	else{
		net->last->next = (node*) malloc(sizeof(node));
		net->last->next->before = net->last;
		createSoftMaxLayer(nin,&(net->last->next->l));
		net->last = net->last->next;
	}	
}
//Functions
void forwardNetwork(ENeuralNetwork *net,EMatrix* in,EMatrix **o){
	node* tmp;
	EMatrix *t;
	t = in;
	for(tmp = net->head;tmp!=NULL;tmp = tmp->next)
		forwardLayer(tmp->l,t,&t);
	*o = t;
}
void backwardNetwork(ENeuralNetwork *net,EMatrix *err){
	EMatrix *e;
	node *tmp;
	e = err;
	for(tmp = net->last;tmp!=NULL;tmp=tmp->before)
		backwardLayer(tmp->l,e,&e);
}
void updateNetwork(ENeuralNetwork *net,double learning_rate){
	node *tmp;
	for(tmp = net->head;tmp!=NULL;tmp = tmp->next)
		updateLayer(tmp->l,learning_rate);
}
void calculateLossGrad(ENeuralNetwork *net, EMatrix *o, EMatrix *t, EMatrix **g){
	int i,j;
	switch(net->loss_type){
		case 0:
		diffMatrix(o,t,g);
		break;
		case 1:
		createMatrix(o->rows,o->cols,g);
		for(i=0;i<o->rows;i++){
			for(j=0;j<o->cols;j++)
				(*g)->data[i][j] = -t->data[0][j]/o->data[i][j];
		}
		break;
		default:
		break;
	}
}
void trainNetwork(ENeuralNetwork *net, EMatrix *in, EMatrix *out,int maxIt,double learning_rate){
	EMatrix *o,*e;
	node *tmp;
	int i;
	for(i = 0;i<maxIt;i++){
		forwardNetwork(net,in,&o);
		calculateLossGrad(net,o,out,&e);
		backwardNetwork(net,e);
		updateNetwork(net,learning_rate);
	}
}