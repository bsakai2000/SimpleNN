#include <cstdio>
#include <math.h>

#define private public
#include "../src/network.hpp"

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

// Ensure that the loss calculations conform to MSE
int check_loss()
{
	Network* n = new Network(1, 2, 2, 4);

	printf("Testing loss function with integer values... ");
	// Generate two vectors to compute loss with
	double vec1[9] = {1, 5, 6, 3, 2, 7, 34, 1, -8};
	double vec2[9] = {-2, 23, 93, 0, -2, 7, 1, 21, 5};
	double out = n->get_loss(vec1, vec2, 9);
	if(out != 532.5)
	{
		printf("%f is the wrong output, should 532.500\n", out);
		return 1;
	}
	printf("Passed\n");

	printf("Testing loss function with double values... ");
	// Chosen by fair dice roll, guaranteed to be random
	double vec3[9] = {-4.171101552, 2.842574601, -2.732273984, -0.1811689881, 0.1015954349, -4.094040275, 3.475649399, 3.38189863, -3.475049535};
	double vec4[9] = {-4.762854364, -4.494467354, 1.241347261, -1.182362005, 1.420223059, -1.433819369, 1.545288678, 2.420261452, -3.009780973};
	out = n->get_loss(vec3, vec4, 9);
	if(out - 4.703193155 > 0.0001 || out - 4.703193155 < -0.001)
	{
		printf("%f is the wrong output, should 4.703\n", out);
		return 1;
	}
	printf("Passed\n");

	delete n;
	return 0;
}

int main()
{
	if(check_forward_propagation())
	{
		printf("Forward Propagation Test failed\n");
	}
	else if (check_loss())
	{
		printf("Loss Function Test failed\n");
	}
	else
	{
		printf("It worked!\n");
	}
}
