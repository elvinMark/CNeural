#include "neural.h"

//Initialize MultiLayer Perceptron Nerual Network
void initMLPNetwork(EMLPNetwork **net){
	*net = (EMLPNetwork*) malloc(sizeof(EMLPNetwork));
}
//Structure the Network
void addMLPInputLayer(EMLPNetwork *net,int nin){
	net->n_prev = nin;
}
void addMLPLayer(EMLPNetwork *net,int nh,int type){
	if(net->head == NULL){
		net->head = (node*) malloc(sizeof(node));
		net->last = net->head;
		createLayer(net->n_prev,nh,type,&(net->head->l));
		net->n_prev = nh;
	}
	else{
		net->last->next = (node*) malloc(sizeof(node));
		net->last->next->before = net->last;
		createLayer(net->n_prev,nh,type,&(net->last->next->l));
		net->last = net->last->next;
		net->n_prev = nh;
	}
}
void addMLPOutputLayer(EMLPNetwork *net,int nout,int type){
	if(net->head == NULL){
		net->head = (node*) malloc(sizeof(node));
		net->last = net->head;
		createLayer(net->n_prev,nout,type,&(net->head->l));
		net->n_prev = nout;
	}
	else{
		net->last->next = (node*) malloc(sizeof(node));
		net->last->next->before = net->last;
		createLayer(net->n_prev,nout,type,&(net->last->next->l));
		net->last = net->last->next;
		net->n_prev = nout;
	}
}
//Functions
void forwardMLPNetwork(EMLPNetwork *net,EMatrix* in,EMatrix **o){
	node* tmp;
	EMatrix *t;
	t = in;
	for(tmp = net->head;tmp!=NULL;tmp = tmp->next)
		forwardLayer(tmp->l,t,&t);
	*o = t;
}
void backwardMLPNetwork(EMLPNetwork *net,EMatrix *err){
	EMatrix *e;
	node *tmp;
	e = err;
	for(tmp = net->last;tmp!=NULL;tmp=tmp->before)
		backwardLayer(tmp->l,e,&e);
}
void updateMLPNetwork(EMLPNetwork *net,double learning_rate){
	node *tmp;
	for(tmp = net->head;tmp!=NULL;tmp = tmp->next)
		updateLayer(tmp->l,learning_rate);
}
void trainMLPNetwork(EMLPNetwork *net, EMatrix *in, EMatrix *out,int maxIt,double learning_rate){
	EMatrix *o,*e;
	node *tmp;
	int i;
	for(i = 0;i<maxIt;i++){
		forwardMLPNetwork(net,in,&o);
		diffMatrix(o,out,&e);
		backwardMLPNetwork(net,e);
		updateMLPNetwork(net,learning_rate);
	}
}

//Initialize Categorical Nerual Network
void initCategoricalNetwork(ECategoricalNetwork **net){
	*net = (ECategoricalNetwork*) malloc(sizeof(ECategoricalNetwork));
}
//Structure the Network
void addCategoricalInputLayer(ECategoricalNetwork *net,int nin){
	net->n_prev = nin;
}
void addCategoricalLayer(ECategoricalNetwork *net,int nh,int type){
	if(net->head == NULL){
		net->head = (node*) malloc(sizeof(node));
		net->last = net->head;
		createLayer(net->n_prev,nh,type,&(net->head->l));
		net->n_prev = nh;
	}
	else{
		net->last->next = (node*) malloc(sizeof(node));
		net->last->next->before = net->last;
		createLayer(net->n_prev,nh,type,&(net->last->next->l));
		net->last = net->last->next;
		net->n_prev = nh;
	}
}
void addCategoricalOutputLayer(ECategoricalNetwork *net){
	if(net->head == NULL){
		net->head = (node*) malloc(sizeof(node));
		net->last = net->head;
		createSoftMaxLayer(net->n_prev,&(net->head->l));
	}
	else{
		net->last->next = (node*) malloc(sizeof(node));
		net->last->next->before = net->last;
		createSoftMaxLayer(net->n_prev,&(net->last->next->l));
		net->last = net->last->next;
	}
}
//Functions
void forwardCategoricalNetwork(ECategoricalNetwork *net,EMatrix* in,EMatrix **o){
	node* tmp;
	EMatrix *t;
	t = in;
	for(tmp = net->head;tmp!=NULL;tmp = tmp->next)
		forwardLayer(tmp->l,t,&t);
	*o = t;
}
void backwardCategoricalNetwork(ECategoricalNetwork *net,EMatrix *err){
	EMatrix *e;
	node *tmp;
	e = err;
	for(tmp = net->last;tmp!=NULL;tmp=tmp->before)
		backwardLayer(tmp->l,e,&e);
}
void updateCategoricalNetwork(ECategoricalNetwork *net,double learning_rate){
	node *tmp;
	for(tmp = net->head;tmp!=NULL;tmp = tmp->next)
		updateLayer(tmp->l,learning_rate);
}
void trainCategoricalNetwork(ECategoricalNetwork *net, EMatrix *in, EMatrix *out,int maxIt,double learning_rate){
	EMatrix *o,*e;
	node *tmp;
	int i;
	for(i = 0;i<maxIt;i++){
		forwardCategoricalNetwork(net,in,&o);
		backwardCategoricalNetwork(net,out);//We are using softmax and cross entropy in the last layer
		updateCategoricalNetwork(net,learning_rate);
	}
}