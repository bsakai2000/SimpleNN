#include "network.hpp"
#include <cmath>
#include <cstdlib>
#include <cstdio>

#define NUM_BATCHES 10000
#define NUM_EXAMPLES 20
#define INPUT_SPREAD 100

// This example should teach our Network object to calculate the square of
// a given double. We provide NUM_BATCHES batches of NUM_EXAMPLES training
// data points, and then output the Network's calculations of ten additional
// test data points
int main()
{
	// Initialize srand to 10 so the output of this example is deterministic
	srand(10);
	setbuf(stdout, NULL);
	Network* n = new Network(1, 3, 10, 1);

	printf("Beginning Training... ");
	// Create two arrays, one for input and one for expected output
	double** input = (double**) malloc(NUM_EXAMPLES * sizeof(double*));
	double** expected_output = (double**) malloc(NUM_EXAMPLES * sizeof(double*));
	for(int i = 0; i < NUM_EXAMPLES; ++i)
	{
		input[i] = (double*) malloc(sizeof(double));
		expected_output[i] = (double*) malloc(sizeof(double));
	}
	
	// Run NUM_BATCHES batches of training data
	for(int i = 0; i < NUM_BATCHES; ++i)
	{
		// Populate the arrays
		for(int j = 0; j < NUM_EXAMPLES; ++j)
		{
			// We want random values from -INPUT_SPREAD to INPUT_SPREAD
			*(input[j]) = (((double) rand()) / RAND_MAX * INPUT_SPREAD * 2) - INPUT_SPREAD;
			*(expected_output[j]) = pow(*(input[j]), 2);
		}

		// Train with our datasets
		n->train(input, expected_output, NUM_EXAMPLES);
	}

	printf("Done!\n");

	// Free our arrays
	for(int i = 0; i < NUM_EXAMPLES; ++i)
	{
		free(input[i]);
		free(expected_output[i]);
	}
	free(input);
	free(expected_output);

	double in;
	double* out;

	// Output 10 example inputs, and their expected vs actual outputs
	for(int i = 0; i < 10; ++i)
	{
		in = (((double) rand()) / RAND_MAX * INPUT_SPREAD * 2) - INPUT_SPREAD;
		out = n->forward_propagate(&in);
		printf("Input: %.4f\tExpected Output: %.4f\tActual Output: %.4f\tDifference: %f\n", in, pow(in, 2), *out, *out - pow(in, 2));
		free(out);
	}

	delete n;

	return 0;
}
