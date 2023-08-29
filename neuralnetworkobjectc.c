#include "neuralnetwork.h"
#include <stdio.h>

static int counter = 0;

void initMyNeuralNet(struct MyNeuralNet* nn, int numOfHiddenNodes, int numOfOutput, int numOfInputs) {
	nn->numOfHiddenNodes = numOfHiddenNodes;
	nn->numOfOutput = numOfOutput;
	nn->numOfInputs = numOfInputs;
	
	nn->hiddenLayer = (double*)calloc(numOfHiddenNodes, sizeof(double));
	nn->hiddenLayerBias = (double*)calloc(numOfHiddenNodes, sizeof(double));
	nn->outputLayer = (double*)calloc(numOfOutput, sizeof(double));
	nn->outputLayerBias = (double*)calloc(numOfOutput, sizeof(double));
	
	nn->hiddenWeights = (double**)malloc(numOfInputs* sizeof(double*));
	for (size_t i = 0; i < numOfInputs; i++){
		nn->hiddenWeights[i] = (double*)malloc(numOfHiddenNodes* sizeof(double));
	}

	nn->outputWeights = (double**)malloc(numOfHiddenNodes* sizeof(double*));
	for (size_t i = 0; i < numOfHiddenNodes; i++) {
		nn->outputWeights[i] = (double*)malloc(numOfOutput* sizeof(double));
	}

	//putting random weights for the first trys...

	for (int i = 0; i < nn->numOfInputs; i++) {
		for (int j = 0; j < numOfHiddenNodes; j++) {
			nn->hiddenWeights[i][j] = (double)randomWeight();
		}
	}

	for (int i = 0; i < numOfHiddenNodes; i++) {
		for (int j = 0; j < numOfOutput; j++) {
			nn->outputWeights[i][j] = (double)randomWeight();
		}
	}
}

int try(struct MyNeuralNet* nn, double** training_inputs, double** training_outputs, int* trainingSetOrder, int trainingSetSize) {
	int returnedValue = 0;
	double learning_rate = 0.1;
	shuffle(trainingSetOrder, trainingSetSize);
	for (int x = 0; x < trainingSetSize; x++) {
		int i = trainingSetOrder[x];

		// Compute hidden layer activation
		for (int j = 0; j < nn->numOfHiddenNodes; j++) {
			double activation = nn->hiddenLayerBias[j];

			for (int k = 0; k < nn->numOfInputs; k++) {
				activation += training_inputs[i][k] * nn->hiddenWeights[k][j];
			}

			nn->hiddenLayer[j] = sigmoid(activation);
		}

		// Compute output layer activation
		for (int j = 0; j < nn->numOfOutput; j++) {
			double activation = nn->outputLayerBias[j];

			for (int k = 0; k < NUM_HIDDEN_NODES; k++) {
				activation += nn->hiddenLayer[k] * nn->outputWeights[k][j];
			}

			nn->outputLayer[j] = sigmoid(activation);
		}

		if (round(nn->outputLayer[0]) == training_outputs[i][0]) {
				printf("Epoch %d, Input: %g %g, Output: %g, Predicted Output: %g, success\n",
					 counter, training_inputs[i][0], training_inputs[i][1], training_outputs[i][0], nn->outputLayer[0]);
				returnedValue = 1;
				counter++;
			}
		else {
			printf("Epoch , Input: %g %g, Output: %g, Predicted Output: %g, failed\n",
				training_inputs[i][0], training_inputs[i][1], training_outputs[i][0], nn->outputLayer[0]);
		}
		
		// Backpropagation

		// Compute change in output weights
		double* deltaOutput = (double*)malloc(nn->numOfOutput * nn->numOfOutput);

		for (int j = 0; j < nn->numOfOutput; j++) {
			double error = (training_outputs[i][j] - nn->outputLayer[j]);
			deltaOutput[j] = error * dSigmoid(nn->outputLayer[j]);
		}

		// Compute change in hidden weights
		double* deltaHidden = (double*)malloc(nn->numOfHiddenNodes * sizeof(double));
		for (int j = 0; j < nn->numOfHiddenNodes; j++) {
			double error = 0.0;
			for (int k = 0; k < nn->numOfOutput; k++) {
				error += deltaOutput[k] * nn->outputWeights[j][k];
			}
			deltaHidden[j] = error * dSigmoid(nn->hiddenLayer[j]);
		}

		// Update output weights and biases
		for (int j = 0; j < nn->numOfOutput; j++) {
			nn->outputLayerBias[j] += deltaOutput[j] * learning_rate;
			for (int k = 0; k < nn->numOfHiddenNodes; k++) {
				nn->outputWeights[k][j] += nn->hiddenLayer[k] * deltaOutput[j] * learning_rate;
			}
		}

		// Update hidden weights and biases
		for (int j = 0; j < nn->numOfHiddenNodes; j++) {
			nn->hiddenLayerBias[j] += deltaHidden[j] * learning_rate;
			for (int k = 0; k < nn->numOfInputs; k++) {
				nn->hiddenWeights[k][j] += training_inputs[i][k] * deltaHidden[j] * learning_rate;
			}
		}
		return returnedValue;
	}
}

void train(struct MyNeuralNet* nn, double** training_inputs, double** training_outputs, int* trainingSetOrder, int trainingSetSize) {
	int numberOfEpochs = 5000;
	for (int epoch = 0; epoch < numberOfEpochs; epoch++) {
		try(nn, training_inputs, training_outputs, trainingSetOrder, trainingSetSize);
	}
}

void trainXOR(struct MyNeuralNet* nn) {
	double** training_inputs = (double**)malloc(NUM_TRAINING_SETS * sizeof(double*));
	double** training_outputs = (double**)malloc(NUM_TRAINING_SETS * sizeof(double*));

	for (int i = 0; i < NUM_TRAINING_SETS; i++) {
		training_inputs[i] = (double*)malloc(NUM_INPUTS * sizeof(double));
		training_outputs[i] = (double*)malloc(NUM_OUTPUTS * sizeof(double));
	}

	// Assign values
	training_inputs[0][0] = 0.0; training_inputs[0][1] = 0.0;
	training_inputs[1][0] = 1.0; training_inputs[1][1] = 0.0;
	training_inputs[2][0] = 0.0; training_inputs[2][1] = 1.0;
	training_inputs[3][0] = 1.0; training_inputs[3][1] = 1.0;

	training_outputs[0][0] = 0.0;
	training_outputs[1][0] = 1.0;
	training_outputs[2][0] = 1.0;
	training_outputs[3][0] = 0.0;

	int trainingSetOrder[] = { 0, 1, 2, 3 };
	int trainingSetSize = 4;
	train(nn, training_inputs, training_outputs, trainingSetOrder, trainingSetSize);
}

int tryXOR(struct MyNeuralNet* nn) {
	double** training_inputs = (double**)malloc(NUM_TRAINING_SETS * sizeof(double*));
	double** training_outputs = (double**)malloc(NUM_TRAINING_SETS * sizeof(double*));

	for (int i = 0; i < NUM_TRAINING_SETS; i++) {
		training_inputs[i] = (double*)malloc(NUM_INPUTS * sizeof(double));
		training_outputs[i] = (double*)malloc(NUM_OUTPUTS * sizeof(double));
	}

	// Assign values
	training_inputs[0][0] = 0.0; training_inputs[0][1] = 0.0;
	training_inputs[1][0] = 1.0; training_inputs[1][1] = 0.0;
	training_inputs[2][0] = 0.0; training_inputs[2][1] = 1.0;
	training_inputs[3][0] = 1.0; training_inputs[3][1] = 1.0;

	training_outputs[0][0] = 0.0;
	training_outputs[1][0] = 1.0;
	training_outputs[2][0] = 1.0;
	training_outputs[3][0] = 0.0;

	int trainingSetOrder[] = { 0, 1, 2, 3 };
	int trainingSetSize = 4;
	int x = try(nn, training_inputs, training_outputs, trainingSetOrder, trainingSetSize);
	return x;
}

void clean(struct MyNeuralNet* nn) {}

