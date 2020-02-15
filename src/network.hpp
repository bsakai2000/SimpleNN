#include <cstdlib>
#include <ctime>
#include <cmath>

class Network
{
	public:
		Network(int num_inputs, int num_layers, int nodes_per_layer, int num_outputs);
		~Network();

		// Get the output of the network given an array of inputs of size num_inputs
		double* forward_propagate(double* input);
		// Train the network using backpropagation, given an array of inputs, an array of outputs,
		// and the size of the two arrays. The arrays should be the same size, and each element of
		// inputs should be of size num_inputs, and each element of expected_outputs should be of
		// size num_outputs
		void train(double** inputs, double** expected_outputs, int num_inputs);
		
	private:
		// The number of nodes in the input layer
		int num_inputs;
		// The number of hidden layers. We expect at least one
		int num_layers;
		// The number of nodes in each hidden layer
		int nodes_per_layer;
		// The number of nodes in the output layer
		int num_outputs;

		// Our set of weights
		// weights[i][j][k] is the weight from node j in layer i to node k in layer i + 1
		double*** weights;

		// Get the loss value between two vectors of length size using Mean Squared Error formula
		double get_loss(double* vector1, double* vector2, int size);
		// Randomly initialize the weights of the network
		void initialize_weights();
		// Our activation function, eg sigmoid
		double get_activation(double x);
		// Update next_layer given the weights and the values of current_layer
		void get_next_layer(double* current_layer, int num_current_layer, double* next_layer, int num_next_layer, double** current_weights);
		// Returns a random number from 0 to 1
		double random_number();
};
