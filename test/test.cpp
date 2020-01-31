#include <cstdio>

#define private public
#include "network.hpp"

// Checks that forward propagation works
// ASSUMES ACTIVATION FUNCTION y(x) = x, NEEDS UPDATING
// IF ACTIVATION FUNCTION CHANGES
int check_forward_propagation()
{
	// Create our network
	Network n(1, 2, 2, 4);

	// Our weights from input to first hidden layer is 1, 2
	for(int i = 0; i < 2; ++i)
	{
		n.weights[0][0][i] = i + 1;
	}

	//Weights between hidden layers are 3, 4, 5, 6
	for(int i = 0; i < 4; ++i)
	{
		n.weights[1][i / 2][i % 2] = i + 3;
	}

	// Weights to output are 7, 8
	for(int i = 0; i < 2; ++i)
	{
		n.weights[2][i][0] = i + 7;
	}

	// Send in 1, and make sure it send back 219
	printf("Testing forward propagation with input 1... ");
	double in = 1;
	double* out = n.forward_propagate(&in);
	if(out[0] != 219)
	{
		printf("%f is the wrong output, should be 219\n", out[0]);
		return 1;
	}
	printf("Passed \n");
	free(out);

	// Send in 3 and make sure it sends back 3 * 219
	printf("Testing forward propagation with input 3... ");
	in = 3;
	out = n.forward_propagate(&in);
	if(out[0] != 657)
	{
		printf("%f is the wrong output, should be 657\n", out[0]);
		return 1;
	}
	printf("Passed \n");
	free(out);

	// Update the weights to include biases, output should be 1712
	n.weights[0][1][0] = 9;
	n.weights[0][1][1] = 10;
	n.weights[1][2][0] = 11;
	n.weights[1][2][1] = 12;
	n.weights[2][2][0] = 13;
	printf("Testing forward propagation and biases with input 1... ");
	in = 1;
	out = n.forward_propagate(&in);
	if(out[0] != 1712)
	{
		printf("%f is the wrong output, should be 1712\n", out[0]);
		return 1;
	}
	printf("Passed \n");
	free(out);

	// Final biases check
	printf("Testing forward propagation and biases with input 3... ");
	in = 3;
	out = n.forward_propagate(&in);
	if(out[0] != 2150)
	{
		printf("%f is the wrong output, should be 2150\n", out[0]);
		return 1;
	}
	printf("Passed \n");
	free(out);
	return 0;
}

int main()
{
	if(check_forward_propagation())
	{
		printf("Forward Propagation Test failed\n");
	}
	else
	{
		printf("It worked!\n");
	}
}
