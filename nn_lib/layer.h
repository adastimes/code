/* Input layer, does not do anything..
 * Author: John ADAS Doe
 * Email: john.adas.doe@gmail.com
 * License: Apache-2.0
 *
 * */
#ifndef _LAYER_
#define _LAYER_

#include <string>
#include <iostream>
#include <fstream>
#include "utils.h"

using namespace std;
template <class T>
class Input { // does not do anything yet

public:
	Input(){

		return;
	}

	void run(Tensor <T> inp, Tensor <T> *out ) {

		return;
	}

	void print()
	{
		cout << "Input\n";
	}

};

#endif
