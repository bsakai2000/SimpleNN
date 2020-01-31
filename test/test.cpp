#include <cstdio>

#define private public
#include "../src/network.hpp"

// Floating point math is hard. taken from https://stackoverflow.com/a/18975072/12432023
// minus the std::
bool double_equals(double a, double b)
{
	double epsilon = 0.00001;
	if(a > b)
	{
		return (a - b) < epsilon;
	}
	return (b - a) < epsilon;
}


// Checks that forward propagation works
// ASSUMES ACTIVATION FUNCTION y(x) = x FOR x > 0,
// NEEDS UPDATING IF ACTIVATION FUNCTION CHANGES
int check_forward_propagation(Network* n)
{
	// Send in 1, and make sure it send back 219
	printf("Testing forward propagation with input 1... ");
	double in = 1;
	double* out = n->forward_propagate(&in);
	if(!double_equals(out[0], 219))
	{
		printf("%f is the wrong output, should be 219\n", out[0]);
		return 1;
	}
	printf("Passed \n");
	free(out);

	// Send in 3 and make sure it sends back 3 * 219
	printf("Testing forward propagation with input 3... ");
	in = 3;
	out = n->forward_propagate(&in);
	if(!double_equals(out[0], 657))
	{
		printf("%f is the wrong output, should be 657\n", out[0]);
		return 1;
	}
	printf("Passed \n");
	free(out);
return 0;
}

int check_biases(Network* n)
{
	// Same as forward propagation checks, but now biases are set
	printf("Testing forward propagation and biases with input 1... ");
	double in = 1;
	double* out = n->forward_propagate(&in);
	if(!double_equals(out[0], 1712))
	{
		printf("%f is the wrong output, should be 1712\n", out[0]);
		return 1;
	}
	printf("Passed \n");
	free(out);

	// Final biases check
	printf("Testing forward propagation and biases with input 3... ");
	in = 3;
	out = n->forward_propagate(&in);
	if(!double_equals(out[0], 2150))
	{
		printf("%f is the wrong output, should be 2150\n", out[0]);
		return 1;
	}
	printf("Passed \n");
	free(out);
	return 0;
}

// Checks that LRELU is leaking properly
int check_leaks(Network* n)
{
	// All nodes will evaluate negative in this test
	printf("Testing forward propagation and biases with input -50... ");
	double in = -50;
	double* out = n->forward_propagate(&in);
	if(!double_equals(out[0], -6.613))
	{
		printf("%f is the wrong output, should be -6.613\n", out[0]);
		return 1;
	}
	printf("Passed \n");
	free(out);

	// Only some nodes will evaluate negative in this test
	printf("Testing forward propagation and biases with input -14... ");
	in = -14;
	out = n->forward_propagate(&in);
	if(!double_equals(out[0], 15.86))
	{
		printf("%f is the wrong output, should be 15.86\n", out[0]);
		return 1;
	}
	printf("Passed \n");
	free(out);
	return 0;
}

int main()
{
	// Create our network
	Network* n = new Network(1, 2, 2, 4);
	
	// Our weights from input to first hidden layer is 1, 2
	for(int i = 0; i < 2; ++i)
	{
		n->weights[0][0][i] = i + 1;
	}

	//Weights between hidden layers are 3, 4, 5, 6
	for(int i = 0; i < 4; ++i)
	{
		n->weights[1][i / 2][i % 2] = i + 3;
	}

	// Weights to output are 7, 8
	for(int i = 0; i < 2; ++i)
	{
		n->weights[2][i][0] = i + 7;
	}

	// No biases yet
	n->weights[0][1][0] = 0;
	n->weights[0][1][1] = 0;
	n->weights[1][2][0] = 0;
	n->weights[1][2][1] = 0;
	n->weights[2][2][0] = 0;

	if(check_forward_propagation(n))
	{
		printf("Forward Propagation Test failed\n");
		return 1;
	}

	// Update the weights to include biases, output should be
	n->weights[0][1][0] = 9;
	n->weights[0][1][1] = 10;
	n->weights[1][2][0] = 11;
	n->weights[1][2][1] = 12;
	n->weights[2][2][0] = 13;

	if(check_biases(n))
	{
		printf("Biases Test failed\n");
		return 1;
	}

	if(check_leaks(n))
	{
		printf("Leaky RELU Test failed\n");
		return 1;
	}

	printf("It worked!\n");
	delete n;
}
