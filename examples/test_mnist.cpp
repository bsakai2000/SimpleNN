#include "network.hpp"
#include <cstdio>
#include <cstdint>

#define NUM_BATCHES 10000
#define NUM_EXAMPLES 20

// Converts a data type between little and big endian. Pass a reference
// to the data as a char* so it can be manipulated like a byte array
void convert(char* value, int value_size)
{
	// Swap bytes so the data is mirrored
	for(int i = 0; i < value_size / 2; ++i)
	{
		char a = value[i];
		value[i] = value[value_size - 1 - i];
		value[value_size - 1 - i] = a;
	}
}

// Read the MNIST labels file and return the number of read labels
int read_labels(char* filename, unsigned char** labels)
{
	// Open the label file and make sure we can open it
	FILE* label_file = fopen(filename, "rb");

	if(label_file == NULL)
	{
		printf("Error opening %s\n", filename);
		return -1;
	}

	// Read the file headers to get magic number and filesize
	uint32_t* leadin = (uint32_t*) malloc(2 * sizeof(int32_t));
	fread(leadin, sizeof(uint32_t), 2, label_file);
	// Convert magic number to little endian
	convert((char*) leadin, sizeof(leadin[0]));
	// Check the magic number
	if(leadin[0] != 2049)
	{
		printf("This doesn't look like an MNIST label file!\n");
		return -1;
	}

	// Get the size of the file and convert to little endian
	uint32_t num_labels = leadin[1];
	convert((char*) &num_labels, sizeof(num_labels));

	// Read the labels from the file
	*labels = (unsigned char*) malloc(num_labels * sizeof(unsigned char));
	fread(*labels, sizeof(unsigned char), num_labels, label_file);

	free(leadin);

	return num_labels;
}

// Read the MNIST images file and return the number of read images
int read_images(char* filename, unsigned char** images, int* num_rows, int* num_columns)
{
	// Open the images file and make sure we can open it
	FILE* image_file = fopen(filename, "rb");

	if(image_file == NULL)
	{
		printf("Error opening %s\n", filename);
		return -1;
	}

	// Read the file headers to get magic numbers, fiesize, and image dimensions
	uint32_t* leadin = (uint32_t*) malloc(4 * sizeof(int32_t));
	fread(leadin, sizeof(uint32_t), 4, image_file);
	// Convert the magic number to little endian and check it
	convert((char*) leadin, sizeof(leadin[0]));
	if(leadin[0] != 2051)
	{
		printf("This doesn't look like an MNIST images file!\n");
		return -1;
	}

	// Get the size of the file and convert to little endian
	uint32_t num_images = leadin[1];
	convert((char*) &num_images, sizeof(num_images));

	// Get the image dimensions and convert to little endian
	convert((char*) (leadin + 2), sizeof(leadin[2]));
	convert((char*) (leadin + 3), sizeof(leadin[3]));
	*num_rows = leadin[2];
	*num_columns = leadin[3];

	// Read the images from the file
	*images = (unsigned char*) malloc(num_images * (*num_rows) * (*num_columns) * sizeof(unsigned char));
	fread(*images, sizeof(unsigned char), num_images * (*num_rows) * (*num_columns), image_file);

	free(leadin);

	return num_images;
}

// Return a single image as a double array instead of a char*
double* get_image(int index, int num_rows, int num_columns, unsigned char* images)
{
	double* image = (double*) malloc(num_rows * num_columns * sizeof(double));
	unsigned char* image_ptr = images + (num_rows * num_columns * index);
	for(int i = 0; i < num_rows * num_columns; ++i)
	{
		image[i] = image_ptr[i];
	}
	return image;
}

// This example should read the MNIST training data to train a Network object,
// and then use that trained Network to guess 10 random images. This example
// is obviously incomplete because we're testing against training data instead
// of against testing data, but it is sufficient for this purpose and could easily
// be adapted to do the full MNIST test as well
int main(int argc, char* argv[])
{
	// Seed the random number generator with 10 for deterministic execution 
	srand(10);
	setbuf(stdout, NULL);
	
	unsigned char* labels;
	unsigned char* images;
	int num_rows, num_columns;

	// We take the labels and images files as arguments
	if(argc != 3)
	{
		printf("This program trains a Neural Network to guess handwritten numbers from the MNIST training set\n");
		printf("Usage: %s <LABELS FILE> <IMAGES FILE>\n", argv[0]);
		return 1;
	}

	printf("Reading Files... ");
	// Read the two MNIST files
	int num_labels = read_labels(argv[1], &labels);
	int num_images = read_images(argv[2], &images, &num_rows, &num_columns);

	// Make sure we read the files correctly
	if(num_labels == -1 || num_images == -1)
	{
		return 1;
	}
	printf("Done!\nBeginning Training... ");

	// Create a Network with an input for each pixel of the image, and an output
	// for each possible value of the image
	Network* n = new Network(num_rows * num_columns, 20, 20, 10);

	// Allocate memory for training purposes
	double** inputs = (double**) malloc(NUM_EXAMPLES * sizeof(double*));
	double** expected_outputs = (double**) malloc(NUM_EXAMPLES * sizeof(double*));
	for(int i = 0; i < NUM_EXAMPLES; ++i)
	{
		expected_outputs[i] = (double*) malloc(10 * sizeof(double));
	}
	// Run NUM_BATCHES batches of NUM_EXAMPLES training data
	for(int i = 0; i < NUM_BATCHES; ++i)
	{
		for(int j = 0; j < NUM_EXAMPLES; ++j)
		{
			// Grab a random image and associated label
			int index = rand() % num_images;
			inputs[j] = get_image(index, num_rows, num_columns, images);
			// Output should be 0 for every number that is incorrect, and 1
			// for the number that is correct
			for(int k = 0; k < 10; ++k)
			{
				expected_outputs[j][k] = 0;
			}
			expected_outputs[j][labels[index]] = 1;
		}
		// Train with our training data
		n->train(inputs, expected_outputs, NUM_EXAMPLES);
		for(int j = 0; j < NUM_EXAMPLES; ++j)
		{
			free(inputs[j]);
		}
	}
	// Free training data
	for(int i = 0; i < NUM_EXAMPLES; ++i)
	{
		free(expected_outputs[i]);
	}
	free(inputs);
	free(expected_outputs);

	printf("Done!\n");

	// Output 10 random training images and what the network thought
	// they were
	for(int i = 0; i < 10; ++i)
	{
		// Grab a random image and associated label
		int index = rand() % num_images;
		double* image = get_image(index, num_rows, num_columns, images);
		double* output = n->forward_propagate(image);
		// The highest number represents the Network's best guess
		int highestIndex = 0;
		double highest = output[0];
		for(int j = 1; j < 10; ++j)
		{
			if(output[j] > highest)
			{
				highest = output[j];
				highestIndex = j;
			}
		}
		// Output the information
		printf("Index: %d\tExpected Output: %d\tActual Output: %d\n", index, labels[index], highestIndex);
		free(image);
		free(output);
	}

	delete n;
	free(labels);
	free(images);
}
