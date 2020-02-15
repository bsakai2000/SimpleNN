#include <cstdio>
#include <math.h>

#define private public
#include "network.hpp"

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

// Checks that the weights are being initialized properly
int check_he_init(Network* n)
{
	// If everything initialized properly these weights should
	// have known values that we can test against
	printf("Testing He Initialized weights with input 10... ");
	double in = 10;
	double* out = n->forward_propagate(&in);
	if(!double_equals(out[0], 12.42613052))
	{
		printf("%f is the wrong output, should be 12.4261\n", out[0]);
		return 1;
	}
	printf("Passed \n");
	free(out);

	// Negative tests to make sure Relu is still leaking
	printf("Testing He Initialized weights with input -8... ");
	in = -8;
	out = n->forward_propagate(&in);
	if(!double_equals(out[0], 0.4191911655))
	{
		printf("%f is the wrong output, should be 0.4191\n", out[0]);
		return 1;
	}
	printf("Passed \n");
	free(out);
	return 0;
}

// Ensure that the loss calculations conform to MSE
int check_loss(Network* n)
{
	printf("Testing loss function with integer values... ");
	// Generate two vectors to compute loss with
	double vec1[9] = {1, 5, 6, 3, 2, 7, 34, 1, -8};
	double vec2[9] = {-2, 23, 93, 0, -2, 7, 1, 21, 5};
	double out = n->get_loss(vec1, vec2, 9);
	if(!double_equals(out, 532.5))
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
	if(!double_equals(out, 4.703193155))
	{
		printf("%f is the wrong output, should 4.703\n", out);
		return 1;
	}
	printf("Passed\n");
	return 0;
}

int main()
{
	// Create our network
	Network* n = new Network(1, 2, 2, 1);

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

	srand(0);
	n->initialize_weights();

	if(check_he_init(n))
	{
		printf("He Initialization Test Failed\n");
		return 1;
	}

	if (check_loss(n))
	{
		printf("Loss Function Test failed\n");
		return 1;
	}

	printf("It worked!\n");
	delete n;
}
