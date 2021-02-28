/* Utilities for our toy deployment environment.
 * Author: John ADAS Doe
 * Email: john.adas.doe@gmail.com
 * License: Apache-2.0
 *
 * */
#ifndef _UTILS_
#define _UTILS_

#include <cfenv>
#include <cmath>
#include <sstream>
#include <string>
#include <iostream>
#include <cassert>
#include <fstream>
#include <typeinfo>
#include <cstdint>
#include <stdio.h>
#include <stdlib.h>


using namespace std;


template <class T>
class  Tensor {


public:
	unsigned int dim[4];
	unsigned int offset1, offset2, offset3;
	unsigned vec_size; // vector size; if the first dimensions are zero we need to make them one for multiplication purposes

	T *arr;

	//constructor
	Tensor(unsigned int dim_in[4]) {

		this->dim[0] = dim_in[0];
		this->dim[1] = dim_in[1];
		this->dim[2] = dim_in[2];
		this->dim[3] = dim_in[3];

		assert(dim[3]!=0);

		this->offset1 = dim_in[1] * dim_in[2] *dim_in[3];
		this->offset2 = dim_in[2]*dim_in[3];
		this->offset3 = dim_in[3];

		this->vec_size = this->dim[0] * this->dim[1] * this->dim[2]* this->dim[3];

		this->arr = new T[this->vec_size];

	}

	T get(unsigned int i, unsigned int j, unsigned int k, unsigned int p)
	{
		return arr[i * this->offset1 + j * this->offset2 + k*this->offset3 + p];
	}

	void reset()// make everything zero
	{
		for(int i=0;i<vec_size;i++)
			arr[i] = 0;
	}

	void print()// print the tensor
	{
		for(unsigned int i=0;i<this->dim[0];i++) {
			for(unsigned int j=0;j<this->dim[1];j++) {
				for(unsigned int k=0;k<this->dim[2];k++) {
					for(unsigned int p=0;p<this->dim[3];p++) {
						cout << this->get(i,j,k,p)<<",";
					}
					cout << "\n";
				}
				cout << "-----------------\n";
			}
			cout << "\n";
		}

	}

	// get the data from the tensor in another tensor
	void get_window(unsigned int j_in,unsigned int  k_in,int width_start, int width_stop, int height_start, int height_stop, Tensor <T> *data)
	{
		int ind_j, ind_k;
		int poz = 0;

		for(unsigned int i=0;i<this->dim[1];i++) {// for all the channels
			for(int j = height_start;j<height_stop;j++){// for the height
				ind_j = j_in + j;
				for(int k=width_start;k<width_stop;k++){// for the width
					ind_k = k_in + k;
					if((ind_j>=0)&&(ind_k>=0)&&(ind_j<(int)this->dim[2])&&(ind_k<(int)this->dim[3])) {

						data->arr[poz] = this->get(0,i,ind_j,ind_k);poz++;
						//printf("(%d,%d,%d) = %f\n", i, ind_j,ind_k,this->get(0,i,ind_j,ind_k));
					}
					else {
						data->arr[poz] = 0;poz++;
					}


				}
			}
		}


	}

	// tensor out = sum(current * in), here we need some carefull gymnastics if we switch from float
	//TODO : need different datatypes for weights, might need to move this to convolution itself to have compute in one place
	T dot_prod(Tensor <T> inp)
	{
		T out = 0;
		assert(this->vec_size==inp.vec_size);

		for(unsigned int i=0;i<this->vec_size;i++)
		{
			out+=this->arr[i]*inp.arr[i];
		}

		return out;
	}

	//write
	void set(unsigned int i, unsigned int j, unsigned int k, unsigned int p, T value)
	{
		arr[i * this->offset1 + j * this->offset2 + k*this->offset3 + p] = value;
	}

	//copy from tensor where the index parses the first dimension
	void copy_subtensor(unsigned int i, Tensor <T> *data)
	{
		unsigned int offset_tmp = i * this->offset1;

		for(unsigned int ind=0;ind<data->vec_size;ind++)
		{
			data->arr[ind]=this->arr[offset_tmp +ind];
		}
	}

	//write a value in the array
	void set(unsigned int i, T value)
	{
			arr[i] = value;
	}

	// load from file
	int LoadFromTxt(string file_name)
	{
		  ifstream params_file;
		  string line;

		  params_file.open(file_name);

		  if (params_file.is_open())
		  {
			  unsigned int len = 0;
			  while ( getline(params_file, line) )
		      {

				  stringstream s(line);
				  float tmp;
				  while(s>>tmp) {
					  this->arr[len] = tmp;
					  len++;
				  }
		      }
			  assert(len==this->vec_size);
		      params_file.close();
		   }
		  else
		  {
			  return 1; //error, file is not open
		  }

		  return 0;
	}

	// load from file
	int SaveToTxt(string file_name)
	{
		  ofstream params_file;
		  string line;

		  params_file.open(file_name);

		  if (params_file.is_open())
		  {

			  for(unsigned int i=0;i<this->vec_size;i++)
				  params_file << this->arr[i]<<"\n";

			  params_file.close();
		  }
		  else
		  {
			  return 1; //error, file is not open
		  }

		  return 0;
	}

	//compare the current tensor against some data from a file
	unsigned int compare(string file_name)
	{
		Tensor <float> vec(this->dim);
		unsigned int err = 0;

		vec.LoadFromTxt(file_name);
		assert(this->vec_size==vec.vec_size);

		for(unsigned int i = 0; i<this->dim[0];i++){
			for(unsigned int j = 0; j<this->dim[1];j++){
				for(unsigned int k = 0; k<this->dim[2];k++){
					for(unsigned int p = 0; p<this->dim[3];p++){

						if(abs(this->get(i,j,k,p) - vec.get(i, j, k, p)) >0.01)
						{
							cout << "("<<i<<","<<j<<","<<k<<","<<p<<")";
							err++;

							if (err >20) return err;
						}

					}
				}
			}
		}

		return err;

	}

};


#endif
