/**
 * @file matrix_generator.cu
 *
 * @brief Matrix generator utility. Generates a random matrix of size N x M.
          Matrix generation is done in parrallel on the GPU using CUDA.
 *
 * @author Abas Farah
 * Contact: contact@mutegalaxy.com
 *
 */

#include "matrix_generator.h"

#define _POSIX_C_SOURCE 199309L
#include <time.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <cuda.h>
#include <curand_kernel.h>
#include <fstream>

using namespace std;

/* Gives us high-resolution timers. */

/**
* @brief Return the number of seconds since an unspecified time (e.g., Unix
*        epoch). This is accomplished with a high-resolution monotonic timer,
*        suitable for performance timing.
*
* @return The number of seconds.
*/
static inline double monotonic_seconds()
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/**
* @brief Output the seconds elapsed while execution.
*
* @param seconds Seconds spent on execution, excluding IO.
*/
static void print_time(double const seconds)
{
  printf("Matrix generation time: %0.04fs\n", seconds);
}

void print_help(){
  cout << "USAGE: matrix_generator [DEST] [N] [M]\n\n";
  cout << "Generate a matrix stored at [DEST] with [N] rows and [M] columns\n\n";
  cout << "  First line of DEST file will contain 'N M'\n";
  cout << "  Followed by N rows of M space seperated integer\n";
}

void printMatrix(vector<int> matrix, int N, int M) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < M; ++j) {
      cout << matrix[i*M+j] << " ";
    }
    cout << endl;
  }
}

// This should generate a unique value for each matrix element
// It should use curand and every element should be unique
__global__ void fillMatrixKernal(int *matrix, int N, int M){
  int i = blockIdx.x;
  int j = threadIdx.x;
  curandState_t state;
  curand_init(clock64(), i, j, &state);
  matrix[i*M+j] = curand(&state) % 100;
}

int main(int argc, char* argv[]) {
  if(argc == 2 && (argv[1] == "-h" || argv[1] == "--help")){
    print_help();
    return 1;
  }else if (argc != 4) {
    cout << "ERROR: Wrong number of arguments.\n\n";
    print_help();
    return 1;
  }

  int N, M;
  string destFile = argv[1];
  N = atoi(argv[2]);
  M = atoi(argv[3]);

  if (N <= 0 || M <= 0) {
    cout << "ERROR: N and M must be greater than 0 and valid integers.\n\n";
    print_help();
    return 1;
  }

  cout << "Generating matrix with " << N << " rows and " << M << " columns" << endl;

  double const start_time = monotonic_seconds();

  // Init cpu matrix
  vector<int> matrix(N*M, 0);

  // Init gpu matrix
  int *d_matrix;
  cudaMalloc(&d_matrix, N*M*sizeof(int));
  cudaMemcpy(d_matrix, matrix.data(), N*M*sizeof(int), cudaMemcpyHostToDevice);

  // Fill gpu matrix with random values
  dim3 blocks(N);
  dim3 threads(M);
  fillMatrixKernal<<<blocks, threads>>>(d_matrix, N, M);

  // Copy gpu matrix to cpu matrix
  cudaMemcpy(matrix.data(), d_matrix, N*M*sizeof(int), cudaMemcpyDeviceToHost);
  printMatrix(matrix, N, M);

  ofstream matrixFile(destFile);

  matrixFile << N << " " << M << endl;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < M; ++j) {
      matrixFile << matrix[i*M+j] << " ";
    }
    matrixFile << endl;
  }
  matrixFile.close();

  double const end_time = monotonic_seconds();
  print_time(end_time - start_time);
}
