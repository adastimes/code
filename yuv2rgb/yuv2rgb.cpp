/**
 * Compute the sum an array
 * @param n number of elements
 * @param array input array
 * @return sum
 */
extern "C" // required when using C++ compiler

void yuv2rgb(int h,int w, int* y, int* u, int* v, int* r, int* g, int* b) {

    int ind_y, ind_uv;
    for (int i = 0; i < h; i++) {
        for(int j=0;j < w; j++) {
           ind_y = i * w + j;
           ind_uv =  (i>>1)*(w>>1) + (j>>1);

           r[ind_y] = y[ind_y] + (1.370705 * (v[ind_uv] - 128));
           g[ind_y] = y[ind_y] + (0.698001 * (v[ind_uv] - 128)) - (0.337633 * (u[ind_uv] - 128));
           b[ind_y] = y[ind_y] + (1.732446 * (u[ind_uv] - 128));

        }
    }

}

