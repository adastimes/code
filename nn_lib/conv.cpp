/* The main function running the network
 * Author: John ADAS Doe
 * Email: john.adas.doe@gmail.com
 * License: Apache-2.0
 *
 * */

#include "conv.h"
#include <iostream>
#include "netw.h"
#include <stdio.h>
#include "utils.h"
#include <cstdint>
#include <time.h>
#include <sys/time.h>

using namespace std;


int main() {

	Network_List <float> list;
	struct timeval start, end;

	list.load_model("/mnt/nvme/model.txt");
	list.start->load_data("/mnt/nvme/image.txt");

	gettimeofday(&start, NULL);
    list.run();
	gettimeofday(&end, NULL);

    printf("Total time %f ms \n", ((float)((end.tv_sec*1000000  + end.tv_usec) - (start.tv_sec*1000000 + start.tv_usec)))/1000000);

    list.current->op_out->SaveToTxt("/mnt/nvme/out.txt");

    //TODO

	//list.summary();

	return 0;

}

