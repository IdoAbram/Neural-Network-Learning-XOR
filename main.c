#include <stdio.h>
#include "neuralnetwork.h"


int main() {

    struct MyNeuralNet nn;
    initMyNeuralNet(&nn, NUM_HIDDEN_NODES, NUM_OUTPUTS, NUM_INPUTS);
    double count = 0;
    double successRate1, successRate2, successRate3;
    for (int i = 0; i < 1000; i++) {
        count += tryXOR(&nn);
    }
    successRate1 = count/10;
    trainXOR(&nn);
    count = 0;
    for (int i = 0; i < 2; i++) {
        trainXOR(&nn);
    }
    for (int i = 0; i < 1000; i++) {
        count += tryXOR(&nn);
    }
    successRate2 = count/10;
    for (int i = 0; i < 2; i++) {
        trainXOR(&nn);
    }
    count = 0;
    for (int i = 0; i < 1000; i++) {
        count += tryXOR(&nn);
    }
    successRate3 = count / 10;
    printf("\nTrying to learn XOR\nBefore training:(basically it is guessing so it is about 50 precentages)\n");
    printf("Success rate: %g precentages\n", successRate1);
    printf("After few training:\nSuccess rate: %g precentages\n",successRate2);
    printf("After more training:\nSuccess rate: %g precentages\n",successRate3);

    clean(&nn);
    return 0;
}