#include "network.hpp"
#include <cstdio>
#include <cstdint>

#define NUM_BATCHES 10000
#define NUM_EXAMPLES 20
#define HEADER_SIZE 138

// Read a BMP file denoted by filename into image_buffer, and set image_height and image_width
// to the values given by the file
int read_image(char* filename, char** image_buffer, int* image_width, int* image_height, char** return_header)
{
	// Open the image file for reading and ensure that it is readable
	FILE* image_file = fopen(filename, "rb");

	if(image_file == NULL)
	{
		printf("Error opening %s\n", filename);
		return 1;
	}

	// Read the BMP header to get necessary information
	char* header = (char*) malloc(HEADER_SIZE * sizeof(char));
	fread(header, sizeof(char), HEADER_SIZE, image_file);
	// Check the magic bytes to ensure we're actually working on a BMP
	if((*(int16_t*) header) != 0x4D42)
	{
		printf("This doesn't look like a BMP file!\n");
		return 1;
	}

	// Retrieve some of the important bits of data from the file
	int file_size = *(int32_t*) (header + 2);
	int data_offset = *(int32_t*) (header + 10);
	int DIB_size = *(int32_t*) (header + 14);

	// We only want to work with BITMAPv5, so check it's not an older version
	if(DIB_size != 124)
	{
		printf("Unrecognized DIB header format! This tool only works on BITMAPV5\n");
		return 1;
	}

	// We only want to work with 24bit color, or RGB. Anything less is boring, anything more is too complicated
	int DIB_bitcount = *(int16_t*) (header + 28);

	if(DIB_bitcount != 24)
	{
		printf("Unrecognized DIB BitCount! This tool only works on 24bit pixel BMPs\n");
	}

	// Grab and store image dimensions
	*image_width = *(int32_t*) (header + 18);
	*image_height = *(int32_t*) (header + 22);

	// Read the image data into the buffer
	*image_buffer = (char*) malloc((file_size - data_offset) * sizeof(char));
	fseek(image_file, data_offset, SEEK_SET);
	fread(*image_buffer, sizeof(char), file_size - data_offset, image_file);

	*return_header = header;

	return 0;
}

// Get a 3 char array of RGB values for the pizel at (x, y) in image_buffer. We return this
// as a double* instead of a char* because we're passing it to the Network
double* get_pixel(int x, int y, char* image_buffer, int image_height, int image_width)
{
	// If the pixel is out of bounds, don't even try
	if(x > image_width || y > image_height)
	{
		return NULL;
	}

	// Allocate space for the pixel
	double* pixel = (double*) malloc(3 * sizeof(double));

	// Calculate the width of a row, aligned to 4 bytes
	int row_width = image_width * 3;
	row_width = ((row_width + 3) / 4) * 4;

	// Read the BGR values into the pixel
	char* pixel_ptr = image_buffer + (y * row_width) + x * 3;
	for(int i = 0; i < 3; ++i)
	{
		pixel[i] = pixel_ptr[i];
	}

	// Return the pixel array
	return pixel;
}

// Generate a BMP pixel array from a trained Neural Network
char* generate_image(Network* n, int image_height, int image_width)
{
	// Calculate the width of a row, aligned to 4 bytes
	int row_width = image_width * 3;
	row_width = ((row_width + 3) / 4) * 4;

	// Allocate space for our image
	char* image_data = (char*) malloc(image_height * row_width * 3 * sizeof(char));
	char* image_ptr = image_data;
	double* input = (double*) malloc(2 * sizeof(double));
	// Loop through each x and y value of the image to calculate its pixel values
	for(int i = 0; i < image_height; ++i)
	{
		for(int j = 0; j < image_width; ++j)
		{
			// Send the Network its coordinates and get the pixel values
			input[0] = j;
			input[1] = i;
			double* pixel = n->forward_propagate(input);
			for(int k = 0; k < 3; ++k)
			{
				// Fix the pixel values to acceptable limits
				double pixelValue = pixel[k];
				if(pixelValue < 0)
				{
					pixelValue = 0;
				}
				if(pixelValue > 255)
				{
					pixelValue = 255;
				}
				// Write the values to the image
				*image_ptr = (char) pixelValue;
				++image_ptr;
			}
			free(pixel);
		}
		// Move image_ptr to the start of the next pixel row
		image_ptr += row_width - image_width;
	}
	return image_data;
}

// This example should teach our network to draw a BMP image by learning to output BGR pixel
// values given x and y coordinates of that pixel. Idea shamelessly stolen from
// https://cs.stanford.edu/people/karpathy/convnetjs/demo/image_regression.html,
int main(int argc, char* argv[])
{
	// Initialize srand to 10 so the output of this example is deterministic
	srand(10);
	setbuf(stdout, NULL);
	Network* n = new Network(2, 5, 10, 3);

	// We only take the image file name as an argument
	if(argc != 2)
	{
		printf("This program trains a Neural Network to rebuild a BMP image given just pixel coordinates\n");
		printf("Usage: %s <BMP FILE>\n", argv[0]);
		return 1;
	}

	// Read in the image, and give up if reading fails
	char* image_buffer;
	char* header;
	int image_width, image_height;
	if(read_image(argv[1], &image_buffer, &image_width, &image_height, &header) == 1)
	{
		return 1;
	}

	// Allocate memory for input and expected output
	double** input = (double**) malloc(NUM_EXAMPLES * sizeof(double*));
	double** expected_output = (double**) malloc(NUM_EXAMPLES * sizeof(double*));
	for(int i = 0; i < NUM_EXAMPLES; ++i)
	{
		input[i] = (double*) malloc(3 * sizeof(double));
	}

	// Run NUM_BATCHES batches of training data
	for(int i = 0; i < NUM_BATCHES; ++i)
	{
		// Populate the arrays
		for(int j = 0; j < NUM_EXAMPLES; ++j)
		{
			int x = rand() % image_width;
			int y = rand() % image_height;
			input[j][0] = x;
			input[j][1] = y;
			expected_output[j] = get_pixel(x, y, image_buffer, image_height, image_width);
		}
		// Train with out datasets
		n->train(input, expected_output, NUM_EXAMPLES);
	}

	// Generate image from the Network
	char* image_data = generate_image(n, image_height, image_width);

	// Output the header
	for(int i = 0; i < HEADER_SIZE; ++i)
	{
		printf("%c", header[i]);
	}
	
	// Calculate the width of a row, aligned to 4 bytes
	int row_width = image_width * 3;
	row_width = ((row_width + 3) / 4) * 4;
	// Output the image data
	for(int i = 0; i < image_height * row_width * 3; ++i)
	{
		printf("%c", image_data[i]);
	}

	return 0;
}
