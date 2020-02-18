#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cstdio>

class Network
{
	public:
		Network(int num_inputs, int num_layers, int nodes_per_layer, int num_outputs)
		{
			// Read the RGB file to get the actual values of the pixels so
			// we can send them to the other code
			FILE* image_file = fopen(filename, "rb");

			if(image_file == NULL)
			{
				printf("MOCK FAILED\n");
				exit(1);
			}

			// Read the image file
			image = (unsigned char*) malloc(height * width * 3 * sizeof(unsigned char));
			fread(image, sizeof(unsigned char), height * width * 3, image_file);
		}

		~Network()
		{
			free(image);
			return;
		}

		double* forward_propagate(double* input)
		{
			// Find the location in the array where that pixel is
			double* pixel = (double*) malloc(3 * sizeof(double));
			unsigned char* pixel_ptr = image + (int) ((input[0] + (height - 1 - input[1]) * width) * 3);

			// Load those values into the pixel and return them
			for(int i = 0; i < 3; ++i)
			{
				pixel[i] = (double) pixel_ptr[2 - i];
			}
			return pixel;
		}

		// No need to change anything when training, we just need to avoid linker errors
		void train(double** inputs, double** expected_outputs, int num_inputs)
		{
			return;
		}
	private:
		unsigned char* image;
		int height = 225;
		int width = 225;
		char filename[17] = "examples/cat.rgb";
};
