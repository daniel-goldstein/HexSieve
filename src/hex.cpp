#include <iostream>
#include <inttypes.h>
#include <omp.h>
#include <ctime>
#include <sys/timeb.h>
#include <stdint.h>
#include <math.h>                   // pow()
#include <cuda_runtime.h>           // cudaFreeHost()
#include "CUDASieve/cudasieve.hpp"  // CudaSieve::getHostPrimes()

//#define billions 1000

unsigned char color_name [] = {'b', 'p', 'r', 'y', 'g', 'c'};
//                            { 0 ,  1 ,  2 ,  3 ,  4 ,  5 }

// Timing functions
double CLOCK() {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC,  &t);
  return (t.tv_sec * 1000)+(t.tv_nsec*1e-6);
}

int shift(int prev, int curr) {
  if(prev % 2 == 0) {
    return (prev + curr) % 6;
  }
  else {
    return (prev - curr + 6) % 6;
  }
}

int main(int argc, char **argv) {

    // run code with argument = number of billions to calculate to
    int billions = atoi(argv[1]);

    int* colors = new int [billions];

    double start, elapsed_time;

    omp_set_num_threads(16);

    start = CLOCK();
    #pragma omp parallel for
    for (uint64_t j = 0; j < billions; j++){
        if (j == 0) printf ("Number of threads: %d\n", omp_get_num_threads());

        uint64_t bottom = j*pow(10,9);
        uint64_t top    = bottom + pow(10,9);
        size_t   len;

        uint64_t * primes = CudaSieve::getHostPrimes(bottom, top, len);

        int color = 0;
        for(uint64_t i = 0; i < len; i++){
          uint64_t currentModulo = primes[i]%6;
          if (primes[i] > 4){
            // TODO Simplify this
            if(currentModulo == 1) {
              if(color % 2 == 0) {
                color = (color + 5) % 6;
              }
              else {
                color = (color + 1) % 6;
              }
            }
            else { // current modulo is equal to 5
              if(color % 2 == 0) {
                color = (color + 1) % 6;
              }
              else {
                color = (color + 5) % 6;
              }
            }
          }
        }
        colors[j] = color;

        // must be freed with this call b/c page-locked memory is used.
        cudaFreeHost(primes);
    }

    elapsed_time = CLOCK() - start;
    printf ("Execution time to calculate %d billions: %f ms\n", billions, elapsed_time);
    printf("%d Billion: %d\n", 1, colors[0]);
    for(uint64_t i = 1; i < billions; i++) {
        colors[i] = shift(colors[i-1], colors[i]);
        printf("%" PRIu64 " Billion: %d\n", i+1, colors[i]);
    }

    delete colors;
    return 0;
}

/*
# mapping numbers to spin color
spin = {  (1,  1) : "blue",
          (2, -1) : "blue",
          (2,  1) : "purple",
          (3, -1) : "purple",
          (3,  1) : "red",
          (4, -1) : "red",
          (4,  1) : "yellow",
          (5, -1) : "yellow",
          (5,  1) : "green",
          (0, -1) : "green",
          (0,  1) : "cyan",
          (1, -1) : "cyan"
        }
*/
