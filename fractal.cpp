/* File:     fractal.cpp
 *
 * Purpose:  compute the Julia set fractals
 *
 * Compile:  g++ -g -Wall -fopenmp -o fractal fractal.cpp -lglut -lGL
 * Run:      ./fractal
 *
 */

#include <iostream>
#include <cstdlib>
#include "../common/cpu_bitmap.h"
#include <omp.h>
using namespace std;

#define DIM 768
#define SERIAL 1
/*Uncomment the following line for visualization of the bitmap*/
#define DISPLAY 1

#define RUNS 10

/* My personal helper functions */

struct cuComplex {
    float   r;
    float   i;
    cuComplex( float a, float b ) : r(a), i(b)  {}
    float magnitude2( void ) { return r * r + i * i; }
    cuComplex operator*(const cuComplex& a) {
        return cuComplex(r*a.r - i*a.i, i*a.r + r*a.i);
    }
    cuComplex operator+(const cuComplex& a) {
        return cuComplex(r+a.r, i+a.i);
    }
};

int julia( int x, int y ) { 
    const float scale = 1.5;
    float jx = scale * (float)(DIM/2 - x)/(DIM/2);
    float jy = scale * (float)(DIM/2 - y)/(DIM/2);

    //cuComplex c(-0.8, 0.156);
    cuComplex c(-0.7269, 0.1889);
    cuComplex a(jx, jy);

    int i = 0;
    for (i=0; i<200; i++) {
        a = a * a + c;
        if (a.magnitude2() > 1000)
            return 0;
    }

    return 1;
}

void kernel_serial ( unsigned char *ptr ){
    for (int y=0; y<DIM; y++) {
        for (int x=0; x<DIM; x++) {
            int offset = x + y * DIM;

            int juliaValue = julia( x, y );
            ptr[offset*4 + 0] = 255 * juliaValue;
            ptr[offset*4 + 1] = 0;
            ptr[offset*4 + 2] = 0;
            ptr[offset*4 + 3] = 255;
        }
    }
}

/*Parallelize the following function using OpenMP*/
void kernel_omp ( unsigned char *ptr ){
    for (int y=0; y<DIM; y++) {
        for (int x=0; x<DIM; x++) {
            int offset = x + y * DIM;

            int juliaValue = julia( x, y );
            ptr[offset*4 + 0] = 255 * juliaValue;
            ptr[offset*4 + 1] = 0;
            ptr[offset*4 + 2] = 0;
            ptr[offset*4 + 3] = 255;
        }
    }
 }

//Question a: 1D rowwise parallel
void kernel_omp_1D_rowwise ( unsigned char *ptr ){
    #pragma omp parallel shared(ptr)
    { 
        int thread_num = omp_get_thread_num();
        int total_threads = omp_get_num_threads();
        for (int y=thread_num; y<DIM; y+=total_threads) {
            for (int x=0; x<DIM; x++) {
                int offset = x + y * DIM;
                int juliaValue = julia( x, y );
                ptr[offset*4 + 0] = 255 * juliaValue;
                ptr[offset*4 + 1] = 0;
                ptr[offset*4 + 2] = 0;
                ptr[offset*4 + 3] = 255;
            }
        }
    }
}

//Question b: 1D columnwise parallel
void kernel_omp_1D_columnwise ( unsigned char *ptr ){
    #pragma omp parallel shared(ptr)
    {
        int thread_num = omp_get_thread_num();
        int total_threads = omp_get_num_threads();
        for (int y=0; y<DIM; y++) {
            for (int x=thread_num; x<DIM; x+=total_threads) {
                int offset = x + y * DIM;
                int juliaValue = julia( x, y );
                ptr[offset*4 + 0] = 255 * juliaValue;
                ptr[offset*4 + 1] = 0;
                ptr[offset*4 + 2] = 0;
                ptr[offset*4 + 3] = 255;
            }
        }
    }
}
 
//Question c: 2D row-block parallel
void kernel_omp_2D_row_block ( unsigned char *ptr ){
    #pragma omp parallel shared(ptr)
    {
        int thread_num = omp_get_thread_num();
        int total_threads = omp_get_num_threads();
        int block_size = DIM / total_threads;
        for (int y=thread_num * block_size; y<DIM; y++) {
            for (int x=0; x<DIM; x++) {
                int offset = x + y * DIM;
                int juliaValue = julia( x, y );
                ptr[offset*4 + 0] = 255 * juliaValue;
                ptr[offset*4 + 1] = 0;
                ptr[offset*4 + 2] = 0;
                ptr[offset*4 + 3] = 255;
            }
            if (((y + 1) % block_size) == 0){
                y += block_size * total_threads;
            }
        }
    }
}

//Question d: 2D column-block parallel
void kernel_omp_2D_column_block ( unsigned char *ptr ){
    #pragma omp parallel shared(ptr)
    {
        int thread_num = omp_get_thread_num();
        int total_threads = omp_get_num_threads();
        int block_size = DIM / total_threads;
        for (int y=0; y<DIM; y++) {
            for (int x=thread_num * block_size; x<DIM; x++) {
                int offset = x + y * DIM;
                int juliaValue = julia( x, y );
                ptr[offset*4 + 0] = 255 * juliaValue;
                ptr[offset*4 + 1] = 0;
                ptr[offset*4 + 2] = 0;
                ptr[offset*4 + 3] = 255;
                if (((x + 1) % block_size) == 0){
                    x += block_size * total_threads;
                }
            }
        }
    }
}

//Question e: OpenMP for construct
void kernel_omp_for_construct( unsigned char *ptr ){
    int y, x;
    #pragma omp parallel for collapse(2) private(y,x) shared(ptr)
    for (y=0; y<DIM; y++) {
        for (x=0; x<DIM; x++) {
            int offset = x + y * DIM;
            int juliaValue = julia( x, y );
            ptr[offset*4 + 0] = 255 * juliaValue;
            ptr[offset*4 + 1] = 0;
            ptr[offset*4 + 2] = 0;
            ptr[offset*4 + 3] = 255;
        }
    }
}
 

void run_benchmark(double finish_s, unsigned char *ptr, void (*function_to_test)(unsigned char *)){
    double start = 0, finish = 0, average_time = 0;
    for (int i=0;i<RUNS;i++){
        start = omp_get_wtime();
        function_to_test(ptr);
        finish += omp_get_wtime() - start;
    }

    average_time = (double) finish / RUNS;
    cout << "Serial time: " << finish_s << endl;
    cout << "Parallel time: " << average_time << endl;
    cout << "Speedup: " << finish_s/average_time << endl;
}

int main( void ) {
    CPUBitmap bitmap( DIM, DIM );
    // Bitmap pointer array for serial
    unsigned char *ptr_s = bitmap.get_ptr();
    unsigned char *ptr_p = bitmap.get_ptr(); 

    double start, finish_s, finish_p; 
    
    /*Serial run*/
    if (SERIAL){
        start = omp_get_wtime();
        kernel_serial( ptr_s );
        finish_s = omp_get_wtime() - start;
    }
    else{
        finish_s = 0.55;
    }
    // Question a:
    cout << "\n----------------------------------------------------------------\n";
    cout << "Benchmarking for 1D rowwise parallel";
    cout << "\n----------------------------------------------------------------\n";
    run_benchmark(finish_s, ptr_p, kernel_omp_1D_rowwise);

    //Question b:
    cout << "\n----------------------------------------------------------------\n";
    cout << "Benchmarking for 1D columwise parallel";
    cout << "\n----------------------------------------------------------------\n";
    run_benchmark(finish_s, ptr_p, kernel_omp_1D_columnwise);

    //Question c:
    cout << "\n----------------------------------------------------------------\n";
    cout << "Benchmarking for 2D row-block parallel";
    cout << "\n----------------------------------------------------------------\n";
    run_benchmark(finish_s, ptr_p, kernel_omp_2D_row_block);

    //Question d:
    cout << "\n----------------------------------------------------------------\n";
    cout << "Benchmarking for 2D column-block parallel";
    cout << "\n----------------------------------------------------------------\n";
    run_benchmark(finish_s, ptr_p, kernel_omp_2D_column_block);

    // Question e:
    cout << "\n----------------------------------------------------------------\n";
    cout << "Benchmarking for OpenMP for contstruct";
    cout << "\n----------------------------------------------------------------\n";
    run_benchmark(finish_s, ptr_p, kernel_omp_for_construct);

    #ifdef DISPLAY     
        bitmap.display_and_exit();
    #endif
}
