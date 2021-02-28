/* Defines the a linked list with the nodes of the neural network. We assume a sequential model here so a linked
 * is fine. Each node will be a layer. There is a function here that loads the mode from a file dumped from
 * python.
 *
 * Author: John ADAS Doe
 * Email: john.adas.doe@gmail.com
 * License: Apache-2.0
 *
 * */

#ifndef _NETW_
#define _NETW_

#include "conv.h"
#include "relu.h"
#include "utils.h"
#include "conv_trans.h"
#include <iostream>
#include <fstream>
#include <string>
using namespace std;

// 0 - Image, 1 - relu, 2 - conv, 3 conv trans
template <class T>
class Network_Node {

	public:
		void *layer;
		Tensor <T> *op_out;
		Network_Node *next;
		unsigned int layer_type;

	Network_Node()
	{
		this->layer = nullptr;
		this->op_out = nullptr;
		this->next = nullptr;
		this->layer_type = -1;
	}

	//Relu and Input nodes
	Network_Node(unsigned int layer_type_in,unsigned int inp_dim[4])
	{
		this->layer_type = layer_type_in;

		if (this->layer_type==1) {

			//printf("Relu\n");

			this->layer = new Relu <T>;

			this->op_out = new Tensor <T>(inp_dim);
		}

		if (this->layer_type==0){ //this is an input node

			//printf("Input\n");
			this->layer = nullptr;
			this->op_out = new Tensor <T>(inp_dim);
		}
		this->next = nullptr;
	}

	//Conv and ConvTrans
	Network_Node(unsigned int layer_type_in,unsigned int stride_in[2], unsigned int padding_in[2], unsigned int kernel_size[2], unsigned int inp_dim[4], unsigned int out_ch)
	{
		unsigned int out_dim[4];

		this->layer_type = layer_type_in;
		if (this->layer_type==2) {

			//printf("Conv2D\n");

			this->layer = new Conv2d <T> (stride_in, padding_in, kernel_size,inp_dim[1], out_ch);

			out_dim[0] = 1;
			out_dim[1] = out_ch;
			out_dim[2] = std::floor(((float)(inp_dim[2] + 2 * padding_in[0] - kernel_size[0]))/stride_in[0] + 1) ;
			out_dim[3] = std::floor(((float)(inp_dim[3] + 2 * padding_in[1] - kernel_size[1]))/stride_in[1] + 1) ;

			this->op_out =  new Tensor <T>(out_dim);

		}

		if (this->layer_type==3) {

			//printf("ConvTrans\n");

			this->layer = new ConvTrans2d <T> (stride_in, padding_in, kernel_size,inp_dim[1], out_ch);

			out_dim[0] = 1;
			out_dim[1] = out_ch;
			out_dim[2] = (inp_dim[2] -1)*stride_in[0] - 2*padding_in[0] +  kernel_size[0];
			out_dim[3] = (inp_dim[3] -1)*stride_in[1] - 2*padding_in[1] +  kernel_size[1];

			this->op_out =  new Tensor <T>(out_dim);
		}
		this->next = nullptr;

	}

	// print the data
	void print()
	{
		if(this->layer_type==0)
			((Input <T>*)layer)->print();

		if(this->layer_type==1)
			((Relu <T> *)layer)->print();

		if(this->layer_type==2)
			((Conv2d <T> *)layer)->print();

		if(this->layer_type==3)
			((ConvTrans2d <T> *)layer)->print();
	}

	void load_params(string file_path, unsigned int id)
	{

		string file_name_bias = file_path + std::to_string(id) + "_bias.txt";
   		string file_name_weights = file_path + std::to_string(id) + ".txt";

		if (this->layer_type==2)//conv
		{
			((Conv2d <T> *)this->layer)->weights->LoadFromTxt(file_name_weights);
			((Conv2d <T> *)this->layer)->bias->LoadFromTxt(file_name_bias);
		}

		if (this->layer_type==3)//conv
		{
			((ConvTrans2d <T> *)this->layer)->weights->LoadFromTxt(file_name_weights);
			((ConvTrans2d <T> *)this->layer)->bias->LoadFromTxt(file_name_bias);
		}
	}

	void load_data(string file_name)
	{
		this->op_out->LoadFromTxt(file_name);
	}

};

template <class T>
class Network_List {

	public:
		Network_Node<T> *start;
		Network_Node<T> *current;
		int len;

		// constructor
		Network_List()
		{
			this->start = nullptr;
			this->current = nullptr;
			this->len = 0;
		}

		// add a node
		void add(Network_Node <T> *node)
		{
			if(this->len==0) {

				this->start = node;
				this->current = node;
			}
			else {
				this->current->next = node;
				this->current = node;
			}
			this->len++;
		}

		T get_node_weights(unsigned int id, unsigned int i, unsigned int j, unsigned int k, unsigned int p)
		{
			Network_Node <T> *node = this->start;

			unsigned int poz = 0;

			while(poz<id)
			{
				node = node->next;
				poz++;
			}

			if (node->layer_type==2) {
				((Conv2d <T> *)(node->layer))->weights->get(i,j,k,p);
			}

			if(node->layer_type==3) {
				((ConvTrans2d <T> *)(node->layer))->weights->get(i,j,k,p);
			}

		}

		T get_node_bias(unsigned int id, unsigned int i, unsigned int j, unsigned int k, unsigned int p)
		{
			Network_Node <T> *node = this->start;

			unsigned int poz = 0;

			while(poz<id)
			{
				node = node->next;
				poz++;
			}

			if (node->layer_type==2) {
				((Conv2d <T> *)(node->layer))->weights->get(i,j,k,p);
			}

			if(node->layer_type==3) {
				((ConvTrans2d <T> *)(node->layer))->weights->get(i,j,k,p);
			}
		}

		//print the layer parameters
		void summary()
		{
			Network_Node <T> *ind = this->start;

			unsigned int id = 0;
			while(ind->next!=nullptr)
			{
				cout << "--------"<<id<<"-----------------\n";
				ind->print();
				cout << "Out_dim = (" << ind->op_out->dim[0] << "," << ind->op_out->dim[1]<<"," << ind->op_out->dim[2]<<"," << ind->op_out->dim[3]<<")\n";
				ind = ind->next;
				id++;
			}
			cout << "--------"<<id<<"-----------------\n";
			ind->print();
			cout << "Out_dim = (" << ind->op_out->dim[0] << "," << ind->op_out->dim[1]<<"," << ind->op_out->dim[2]<<"," << ind->op_out->dim[3]<<")\n";
			cout << "\n";
		}

		// load model and the wrights
		int load_model(string file_name)
		{
			 ifstream params_file;
			 string line;

			 std::size_t found = file_name.find_last_of("/");
			 string file_path = file_name.substr(0,found+1);


			  params_file.open(file_name);


			  if (params_file.is_open())
			  {
			      while ( getline(params_file, line) )
			      {
			        if (line.compare(0,10,"Layer_Type")==0)//we have a new layer here so we can start populating things
			        {
			        	if (line.compare(13,5,"Image")==0)
			        	{
			        		unsigned int inp_dim[4] = {1,0,0,0};

			        		//out_channels
			        		getline(params_file, line);
			        		int nr_num = line.length() - 11;
			        		inp_dim[1] = stoi(line.substr(11,nr_num));

			        		//height
			        		getline(params_file, line);
			        		nr_num = line.length() - 9;
			        		inp_dim[2] = stoi(line.substr(9,nr_num));
			        		//width
			        		getline(params_file, line);
			        		nr_num = line.length() - 8;
			        		inp_dim[3] = stoi(line.substr(8,nr_num));

			        		getline(params_file, line);
			        		Network_Node <T> * node = new Network_Node <T> (0, inp_dim);

			        		this->add(node);
			        		this->len ++;

			        	}

			        	if (line.compare(13,6,"Conv2d")==0)
			        	{
			        		unsigned int id, in_ch, out_ch;
			        		unsigned kernel_size[2];
			        		unsigned stride[2];
			        		unsigned padding[2];


			        		//get nr input channels
			        		getline(params_file, line);//ID
			        		int nr_num = line.length() - 5;
			        		id = stoi(line.substr(5,nr_num));
			        		//cout << "ID = " << id<<"\n";

			        		//in_channels
			        		getline(params_file, line);
			        		nr_num = line.length() - 14;
			        		in_ch = stoi(line.substr(14,nr_num));
			        		assert(in_ch==this->current->op_out->dim[1]);

			        		//out_channels
			        		getline(params_file, line);
			        		nr_num = line.length() - 15;
			        		out_ch = stoi(line.substr(15,nr_num));

			        		//kernel_size
			        		getline(params_file, line);
			        		kernel_size[0] = stoi(line.substr(15,1));
			        		kernel_size[1] = stoi(line.substr(17,1));

			        		//padding
			        		getline(params_file, line);
			        		padding[0]  = stoi(line.substr(11,1));
			        		padding[1]  = stoi(line.substr(13,1));

			        		//stride
			        		getline(params_file, line);
			        		stride[0]  = stoi(line.substr(10,1));
			        		stride[1]  = stoi(line.substr(12,1));

			        		getline(params_file, line);// read -------------------

			        		Network_Node <T> * node;
			        		node = new Network_Node <T> (2, stride, padding, kernel_size, this->current->op_out->dim, out_ch);

			           		node->load_params(file_path, id);
			        		this->add(node);
			        		this->len++;

			        	}

			        	if (line.compare(13,4,"ReLU")==0)
			        	{
			        		Network_Node <T> *node ;
			        		node = new Network_Node <T>(1,this->current->op_out->dim);

			        		this->add(node);
			        		this->len++;

			        	}

			        	if (line.compare(13,15,"ConvTranspose2d")==0)
			        	{

			        		unsigned int id, in_ch, out_ch;
			        		unsigned kernel_size[2];
			        		unsigned stride[2];
			        		unsigned padding[2];


			        		getline(params_file, line);//ID
			        		int nr_num = line.length() - 5;
			        		id = stoi(line.substr(5,nr_num));
			        		//cout << "ID = " << id<<"\n";

			        		//in_channels
			        		getline(params_file, line);
			        		nr_num = line.length() - 14;
			        		in_ch = stoi(line.substr(14,nr_num));
			        		assert(in_ch==this->current->op_out->dim[1]);

			        		//out_channels
			        		getline(params_file, line);
			        		nr_num = line.length() - 15;
			        		out_ch = stoi(line.substr(15,nr_num));

			        		//kernel_size
			        		getline(params_file, line);
			        		kernel_size[0] = stoi(line.substr(15,1));
			        		kernel_size[1] = stoi(line.substr(17,1));

			        		//padding
			        		getline(params_file, line);
			        		padding[0]  = stoi(line.substr(11,1));
			        		padding[1]  = stoi(line.substr(13,1));

			        		//stride
			        		getline(params_file, line);
			        		stride[0]  = stoi(line.substr(10,1));
			        		stride[1]  = stoi(line.substr(12,1));

			        		getline(params_file, line);// read -------------------


			        		Network_Node <T> * node;

			        		node = new Network_Node <T>(3, stride, padding, kernel_size, this->current->op_out->dim, out_ch);
			        		node->load_params(file_path, id);
			        		this->add(node);
			        		this->len++;

			        	}
			        }

			      }

			      params_file.close();
			   }
			  else{
				  return 1;
			  }

			  params_file.close();
			  return 0;
		}

		void run(){

			Network_Node <T> *ind = this->start;
			int id = 0;

#if 0
			int deconv_id=0,conv_id=0;
			string file_path = "/mnt/nvme/";

#endif

			while(ind->next!=nullptr)
			{
				if(ind->next->layer_type==2) { // if next layer is convolution we can run it

					((Conv2d <T> *)ind->next->layer)->run(ind->op_out, ind->next->op_out);

#if 0
					conv_id++;
					string ref_file_name= file_path + "conv" +std::to_string(conv_id) + ".txt";
					cout <<  ind->next->op_out->compare(ref_file_name)<<"\n";
					cout << "----------------\n";
#endif

				}


				if(ind->next->layer_type==3) { // if next layer is convolution we can run it

					((ConvTrans2d <T> *)ind->next->layer)->run(ind->op_out, ind->next->op_out);

#if 0
					deconv_id++;
					string ref_file_name= file_path + + "dconv" + std::to_string(deconv_id) + ".txt";
					cout <<  ind->next->op_out->compare(ref_file_name)<<"\n";
					cout << "----------------\n";
#endif
				}

				if(ind->next->layer_type==1) { // if next layer is convolution we can run it

					((Relu <T> *)ind->next->layer)->run(*(ind->op_out), ind->next->op_out);
				}

				ind = ind->next;
				id++;
			}
		}

};

#endif
