/* Convolution Layer.
 * Author: John ADAS Doe
 * Email: john.adas.doe@gmail.com
 * License: Apache-2.0
 *
 * */

#ifndef _CONV_LAYER_
#define _CONV_LAYER_

#include "layer.h"
#include <cfenv>
#include <cmath>
#include <sstream>
#include <string>
#include <iostream>
#include "utils.h"


template <class T>
class Conv2d {

public:
	unsigned int stride[2];//(x,y)
	unsigned int padding[2];//(x,y)

	Tensor <T> *bias;
	Tensor <T> *weights;

	// initialized the parameters
	Conv2d(unsigned int stride_in[2], unsigned int padding_in[2], unsigned int kernel_size[2], unsigned int in_ch, unsigned int out_ch) {

		this->stride[0] = stride_in[0];this->stride[1] = stride_in[1];
		this->padding[0] = padding_in[0];this->padding[1] = padding_in[1];

		unsigned int weights_dim[4];//[out_ch, in_ch, height, width]
		unsigned int bias_dim[4];//[out_ch]

		weights_dim[0] = out_ch;
		weights_dim[1] = in_ch;
		weights_dim[2] = kernel_size[0];
		weights_dim[3] = kernel_size[1];

		bias_dim[0] = 1;
		bias_dim[1] = 1;
		bias_dim[2] = 1;
		bias_dim[3] = out_ch;

		this->bias = new Tensor <T> (bias_dim);
		this->weights = new Tensor <T> (weights_dim);
	}

	// run the layer; take the input from previous layer and output in the current node, tensors are in nodes
	void run(Tensor <T> *inp, Tensor <T> *out ) {

		int width_start, width_stop;
		int height_start,height_stop;

		unsigned int j_in,k_in;//index in the input tensor
		unsigned int window_dim[4];

		window_dim[0] =1;
		window_dim[1] =this->weights->dim[1];
		window_dim[2] =this->weights->dim[2];
		window_dim[3] =this->weights->dim[3];

		//get the shit around
		width_start = -std::floor(this->weights->dim[3]/2);
		width_stop = this->weights->dim[3] + width_start;
		height_start = -std::floor(this->weights->dim[2]/2);
		height_stop = this->weights->dim[2] + height_start;


		Tensor <float> *data = new Tensor <float>(window_dim);
		Tensor <float> *weights_tmp = new Tensor <float>(window_dim);

			// for each sample in the output
			for(unsigned int j = 0;j<out->dim[2];j++){ // for height
				for(unsigned int k = 0;k<out->dim[3];k++){ // for width

						// where to get the inputs from, if the padding is too small than we move the window a little more in
						j_in = (-height_start-this->padding[0]) + j * this->stride[0];
						k_in = (-width_start-this->padding[1]) + k * this->stride[1];

						inp->get_window(j_in, k_in, width_start, width_stop, height_start, height_stop, data);


						for(unsigned int i=0;i<out->dim[1];i++) { // we reuse the read and multiply with the kernel

							this->weights->copy_subtensor(i, weights_tmp);

							//weights_tmp->print();

							out->set(0, i, j, k, this->bias->arr[i] + data->dot_prod(*weights_tmp));
							//cout << data->dot_prod(*weights_tmp)<<", ";
						}
				}
			}

	}

	//print layer parameters
	void print()
	{
		cout << "Conv2d\n";
		cout << "stride = ( " << this->stride[0]<<","<<this->stride[1]<<")\n";
		cout << "padding = ( " << this->stride[0]<<","<<this->stride[1]<<")\n";
		cout << "kernel_size = ( " << this->weights->dim[0]<<","<<this->weights->dim[1]<<","<<this->weights->dim[2]<<","<<this->weights->dim[3]<<")\n";
		cout << "bias = ( " << this->bias->dim[0]<<","<<this->bias->dim[1]<<","<<this->bias->dim[2]<<","<<this->bias->dim[3]<<")\n";
	}

};

#endif
