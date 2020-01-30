#include "network.hpp"

Network::Network(int num_inputs, int num_layers, int nodes_per_layer, int num_outputs)
{
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

void Network::initialize_weights()
{
	/* TODO */
}

double Network::get_activation(double x)
{
	/* TODO */
	return x;
}
