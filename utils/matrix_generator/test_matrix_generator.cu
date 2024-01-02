#include "matrix_generator.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <fstream>

using namespace std;

void printMatrix(vector<int> matrix, int N, int M) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++)
      cout << matrix[i*M + j] << " ";
    cout << "\n";
  }
}

void print_help(){
  cout << "USAGE: matrix_generator [DEST] [N] [M]\n\n";
  cout << "Generate a matrix stored at [DEST] with [N] rows and [M] columns\n\n";
  cout << "  First line of DEST file will contain 'N M'\n";
  cout << "  Followed by N rows of M space seperated integer\n";
}

int main (int argc, char *argv[]) {
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
  vector<int> matrix = matrix_generator(N, M);

  ofstream matrixFile(destFile);

  matrixFile << N << " " << M << endl;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < M; ++j) {
      matrixFile << matrix[i*M+j] << " ";
    }
    matrixFile << endl;
  }
  matrixFile.close();

  return 0;
}
