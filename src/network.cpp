#include "network.hpp"

Network::Network(int num_inputs, int num_layers, int nodes_per_layer, int num_outputs)
{
	// Check if rand has been seeded. If the caller didn't call srand yet
	// it's important we call it so our random numbers are truly random.
	// If the caller already called srand, assume they did it for a reason
	// and did it correctly. This can be useful for debugging if you want
	// your results to be repeatable
	if(rand() == 1804289383)
	{
		// If rand hasn't been seeded, seed with current time
		srand(time(NULL));
	}

	// Initialize the global variables for our object
	this->num_inputs = num_inputs;
	this->num_layers = num_layers;
	this->nodes_per_layer = nodes_per_layer;
	this->num_outputs = num_outputs;

	// Allocate the weights array
	weights = (double***) malloc((num_layers + 1) * sizeof(double**));

	// Allocate the weights from the input layer to the first hidden layer,
	// remembering to add a virtual input for the bias
	weights[0] = (double**) malloc((num_inputs + 1) * sizeof(double*));
	for(int i = 0; i < (num_inputs + 1); ++i)
	{
		weights[0][i] = (double*) malloc(nodes_per_layer * sizeof(double));
	}

	// Allocate the weights between the hidden layers, again accounting for
	// biases
	for(int i = 1; i < num_layers; ++i)
	{
		weights[i] = (double**) malloc((nodes_per_layer + 1) * sizeof(double*));
		for(int j = 0; j < (nodes_per_layer + 1); ++j)
		{
			weights[i][j] = (double*) malloc(nodes_per_layer * sizeof(double));
		}
	}

	// Allocate the weights from the last hidden layer to the output layer,
	// again accounting for biases
	weights[num_layers] = (double**) malloc((nodes_per_layer + 1) * sizeof(double*));
	for(int i = 0; i < (nodes_per_layer + 1); ++i)
	{
		weights[num_layers][i] = (double*) malloc(num_outputs * sizeof(double));
	}

	initialize_weights();
}

Network::~Network()
{
	// Free the input weights
	for(int i = 0; i < num_inputs + 1; ++i)
	{
		free(weights[0][i]);
	}

	free(weights[0]);

	// Free the hidden layer weights
	for(int i = 1; i < num_layers + 1; ++i)
	{
		for(int j = 0; j < nodes_per_layer + 1; ++j)
		{
			free(weights[i][j]);
		}

		free(weights[i]);
	}

	// Finally, free weights
	free(weights);
}

double* Network::forward_propagate(double* input)
{
	// Allocate a temporary array for node values
	double** nodes = (double**) calloc(num_layers, sizeof(double*));
	for(int i = 0; i < num_layers; ++i)
	{
		nodes[i] = (double*) calloc(nodes_per_layer, sizeof(double));
	}

	// Get the values of the first hidden layer
	get_next_layer(input, num_inputs, nodes[0], nodes_per_layer, weights[0]);
	
	// Propagate through the hidden layers
	for(int i = 1; i < num_layers; ++i)
	{
		get_next_layer(nodes[i - 1], nodes_per_layer, nodes[i], nodes_per_layer, weights[i]);
	}

	// Allocate an output array. Caller is responsible for freeing this.
	double* outputs = (double*) calloc(num_outputs, sizeof(double));
	// Get the values of the output array
	get_next_layer(nodes[num_layers - 1], nodes_per_layer, outputs, num_outputs, weights[num_layers]);

	// Free the nodes arraya
	for(int i = 0; i < num_layers; ++i)
	{
		free(nodes[i]);
	}
	free(nodes);

	return outputs;
}

void Network::get_next_layer(double* current_layer, int num_current_layer, double* next_layer, int num_next_layer, double** current_weights)
{
	// Zero out the next layer
	for(int i = 0; i < num_next_layer; ++i)
	{
		next_layer[i] = 0;
	}

	// Propagate values forward by multiplying the current node by its weight
	// and adding to the next node
	for(int i = 0; i < num_current_layer; ++i)
	{
		for(int j = 0; j < num_next_layer; ++j)
		{
			next_layer[j] += current_layer[i] * current_weights[i][j];
		}
	}

	// Add in the biases
	for(int i = 0; i < num_next_layer; ++i)
	{
		next_layer[i] += current_weights[num_current_layer][i];
	}

	// Use the activation function to get our final values for the nodes
	for(int i = 0; i < num_next_layer; ++i)
	{
		next_layer[i] = get_activation(next_layer[i]);
	}
}

// We use He initialization, an algorithm designed
// to work well with RELU and LRELU
// source https://towardsdatascience.com/weight-initialization-techniques-in-neural-networks-26c649eb3b78
void Network::initialize_weights()
{
	double he_coefficient = sqrt(2.0 / num_inputs);
	// Initialize the weights from input to first layer
	for(int i = 0; i < num_inputs + 1; ++i)
	{
		for(int j = 0; j < nodes_per_layer; ++j)
		{
			weights[0][i][j] = random_number() * he_coefficient; 
		}
	}

	// Initialize the hidden weights
	he_coefficient = sqrt(2.0 / nodes_per_layer);
	for(int i = 1; i < num_layers; ++i)
	{
		for(int j = 0; j < nodes_per_layer + 1; ++j)
		{
			for(int k = 0; k < nodes_per_layer; ++k)
			{
				weights[i][j][k] = random_number() * he_coefficient;
			}
		}
	}

	// Initialize the weights from the last layer to the output
	for(int i = 0; i < nodes_per_layer + 1; ++i)
	{
		for(int j = 0; j < num_outputs; ++j)
		{
			weights[num_layers][i][j] = random_number() * he_coefficient;
		}
	}
}

double Network::random_number()
{
	return ((double) rand()) / RAND_MAX;
}

// We use a Leaky RELU activation
double Network::get_activation(double x)
{
	if(x < 0)
	{
		return x * 0.1;
	}
	return x;
}

double Network::get_loss(double* vector1, double* vector2, int size)
{
	double result = 0;
	// Sum the squares of the differences of the elements of the vectors
	for(int i = 0; i < size; ++i)
	{
		result += pow(vector1[i] - vector2[i], 2);
	}
	// Loss is sum of squares divided by 2 * size
	return result / (2.0 * size);
}

void Network::train(double** inputs, double** expected_outputs, int num_inputs)
{
	/* TODO */
}

double* Network::dump_weights()
{
	// Allocate space for all of the weights and biases
	int num_weights = (num_inputs + 1) * nodes_per_layer
		+ (nodes_per_layer + 1) * nodes_per_layer * (num_layers - 1)
		+ (nodes_per_layer + 1) * num_outputs;
	double* linear_weights = (double*) malloc(num_weights * sizeof(double));
	double* weights_ptr = linear_weights;

	// Read all of the weights from inputs to the first hidden layer
	for(int i = 0; i < num_inputs + 1; ++i)
	{
		for(int j = 0; j < nodes_per_layer; ++j)
		{
			*weights_ptr = weights[0][i][j];
			++weights_ptr;
		}
	}

	// Read all of the weights inside the hidden layers
	for(int i = 0; i < num_layers - 1; ++i)
	{
		for(int j = 0; j < nodes_per_layer + 1; ++j)
		{
			for(int k = 0; k < nodes_per_layer; ++k)
			{
				*weights_ptr = weights[i + 1][j][k];
				++weights_ptr;
			}
		}
	}

	// Read all of the weights from the last hidden layer to the output
	for(int i = 0; i < nodes_per_layer + 1; ++i)
	{
		for(int j = 0; j < num_outputs; ++j)
		{
			*weights_ptr = weights[num_layers][i][j];
			++weights_ptr;
		}
	}

	// Return the weights and biases. The caller must free
	return linear_weights;
}

void Network::load_weights(double* linear_weights)
{
	double* weights_ptr = linear_weights;

	// Read all of the weights from inputs to the first hidden layer
	for(int i = 0; i < num_inputs + 1; ++i)
	{
		for(int j = 0; j < nodes_per_layer; ++j)
		{
			weights[0][i][j] = *weights_ptr;
			++weights_ptr;
		}
	}

	// Read all of the weights inside the hidden layers
	for(int i = 0; i < num_layers - 1; ++i)
	{
		for(int j = 0; j < nodes_per_layer + 1; ++j)
		{
			for(int k = 0; k < nodes_per_layer; ++k)
			{
				weights[i + 1][j][k] = *weights_ptr;
				++weights_ptr;
			}
		}
	}

	// Read all of the weights from the last hidden layer to the output
	for(int i = 0; i < nodes_per_layer + 1; ++i)
	{
		for(int j = 0; j < num_outputs; ++j)
		{
			weights[num_layers][i][j] = *weights_ptr;
			++weights_ptr;
		}
	}
}
