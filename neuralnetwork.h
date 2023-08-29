#pragma once
#include <math.h>
#include <stdlib.h> 


#define NUM_INPUTS 2
#define NUM_HIDDEN_NODES 2
#define NUM_OUTPUTS 1
#define NUM_TRAINING_SETS 4

struct MyNeuralNet
{
	int numOfHiddenNodes, numOfOutput, numOfTrainingSet, numOfInputs;
	double* hiddenLayer, * outputLayer;
	double* hiddenLayerBias, * outputLayerBias;
	double** hiddenWeights, ** outputWeights;
};

void initMyNeuralNet(struct MyNeuralNet* nn, int numOfHiddenNodes, int numOfOutput, int numOfInputs);
int try(struct MyNeuralNet* nn, double** training_inputs, double** training_outputs, int* trainingSetOrder, int trainingSetSize);
void train(struct MyNeuralNet* nn, double** training_inputs, double** training_outputs, int* trainingSetOrder, int trainingSetSize);
void trainXOR(struct MyNeuralNet* nn);
int tryXOR(struct MyNeuralNet* nn);
void clean(struct MyNeuralNet* nn);

double randomWeight();
double sigmoid(double x);
double dSigmoid(double x);
void shuffle(int* array, size_t n);