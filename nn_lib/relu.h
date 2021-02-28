/* Relu Layer.
 * Author: John ADAS Doe
 * Email: john.adas.doe@gmail.com
 * License: Apache-2.0
 *
 * */

#ifndef _RELU_LAYER_
#define _RELU_LAYER_

#include <algorithm>

#define MAX(a, b) ((a) > (b) ? (a) : (b))

template <class T>
class Relu {

public:
	Relu(){

	}

	void run(Tensor <T> inp, Tensor <T> *out ) {

		assert(inp.vec_size==out->vec_size);
		for(unsigned int i =0;i<inp.vec_size;i++)
			out->arr[i] = MAX(0,inp.arr[i]);

	}

	void print()
	{
		cout << "Relu\n";
	}
};

#endif
