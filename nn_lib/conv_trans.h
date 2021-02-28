/* Convolution Transpose Layer.
 * Author: John ADAS Doe
 * Email: john.adas.doe@gmail.com
 * License: Apache-2.0
 *
 * */
#ifndef _CONV_TRANS_LAYER_
#define _CONV_TRANS_LAYER_

#include "layer.h"
#include <cfenv>
#include <cmath>


//careful here the order of the weights is different cudos to pytorch
template <class T>
class ConvTrans2d {

public:
	unsigned int stride[2];//(x,y)
	unsigned int padding[2];//(x,y)

	Tensor <T> *bias;
	Tensor <T> *weights;//[in_ch, out_ch, height, width], the weights are mirrored.. mother fuckers

	//The constructor
	ConvTrans2d(unsigned int stride_in[2], unsigned int padding_in[2], unsigned int kernel_size[2], unsigned int in_ch, unsigned int out_ch) {

		this->stride[0] = stride_in[0];this->stride[1] = stride_in[1];
		this->padding[0] = padding_in[0];this->padding[1] = padding_in[1];

		unsigned int weights_dim[4];//[in_ch, out_ch, height, width]
		unsigned int bias_dim[4];//[out_ch]

		weights_dim[0] = in_ch;
		weights_dim[1] = out_ch;
		weights_dim[2] = kernel_size[0];
		weights_dim[3] = kernel_size[1];

		bias_dim[0] = 1;
		bias_dim[1] = 1;
		bias_dim[2] = 1;
		bias_dim[3] = out_ch;

		this->bias = new Tensor <T> (bias_dim);
		this->weights = new Tensor <T> (weights_dim);
	}

	// get the data from the tensor in another tensor
	void get_window(Tensor <T> *inp, unsigned int j_in,unsigned int  k_in,int width_start, int width_stop, int height_start, int height_stop, Tensor <T> *data)
	{
		int ind_j, ind_k;
		int poz = 0;

		for(unsigned int i=0;i<inp->dim[1];i++) {// for all the channels
			for(int j = height_start;j<height_stop;j++){// for the height
				ind_j = j_in + j;
				for(int k=width_start;k<width_stop;k++){// for the width
					ind_k = k_in + k;

					if((ind_j>=0)&&(ind_k>=0)&&(ind_j<(int)(inp->dim[2]*this->stride[0]))&&(ind_k<(int)(inp->dim[3]*this->stride[1]))) {

						if ((ind_j%this->stride[0]==0) && (ind_k%this->stride[1]==0))
						{
							data->arr[poz] = inp->get(0,i,ind_j/this->stride[0],ind_k/this->stride[1]);
						}
						else {
							data->arr[poz] = 0;
						}
						//printf("(%d,%d,%d) = %f\n", i, ind_j,ind_k,this->get(0,i,ind_j,ind_k));
					}
					else {
						data->arr[poz] = 0;
					}

					poz++;


				}
			}
		}


	}

	//copy from tensor where the index parses the second dimension
	void copy_weights(unsigned int ind, Tensor <T> *out)//[1, in_ch, height, width]
	{

		for(unsigned int i=0;i<this->weights->dim[0];i++){//inp channel
			for(unsigned int k=0;k<this->weights->dim[2];k++){
				for(unsigned int p=0;p<this->weights->dim[3];p++){

					out->set(0,i,k,p,this->weights->get(i,ind,this->weights->dim[2] - k - 1,this->weights->dim[2] - p -1));

				}
			}
		}

	}

	// run the convolution transposed
	void run(Tensor <T> *inp, Tensor <T> *out ) {


		int width_start, width_stop;
		int height_start,height_stop;

		unsigned int j_in,k_in;//index in the input tensor
		unsigned int window_dim[4];

		window_dim[0] =1;
		window_dim[1] =this->weights->dim[0];
		window_dim[2] =this->weights->dim[2];
		window_dim[3] =this->weights->dim[3];

		//get the shit around
		height_start = -(this->weights->dim[2] - this->padding[0]-1);
		height_stop  = this->padding[0]+1;

		width_start = -(this->weights->dim[3] - this->padding[1]-1);
		width_stop = this->padding[1]+1;


		Tensor <float> *data = new Tensor <float>(window_dim);
		Tensor <float> *weights_tmp = new Tensor <float>(window_dim);

			// for each sample in the output
			for(unsigned int j = 0;j<out->dim[2];j++){ // for height
				for(unsigned int k = 0;k<out->dim[3];k++){ // for width

						this->get_window(inp, j, k, width_start, width_stop, height_start, height_stop, data);
						//cout << "("<<j<<","<<k<<")\n";

						for(unsigned int i=0;i<out->dim[1];i++) { // we reuse the read and multiply with the kernel

							this->copy_weights(i, weights_tmp);

							//weights_tmp->print();

							out->set(0, i, j, k, this->bias->arr[i] + data->dot_prod(*weights_tmp));
							//cout << this->bias->arr[i] + data->dot_prod(*weights_tmp)<<"\n ";
						}
				}
			}
	}

	//print the parameters of the layer
	void print()
	{
		cout << "ConvTrans2d\n";
		cout << "stride = ( " << this->stride[0]<<","<<this->stride[1]<<")\n";
		cout << "padding = ( " << this->stride[0]<<","<<this->stride[1]<<")\n";
		cout << "kernel_size = ( " << this->weights->dim[0]<<","<<this->weights->dim[1]<<","<<this->weights->dim[2]<<","<<this->weights->dim[3]<<")\n";
		cout << "bias = ( " << this->bias->dim[0]<<","<<this->bias->dim[1]<<","<<this->bias->dim[2]<<","<<this->bias->dim[3]<<")\n";
	}


};

#endif
