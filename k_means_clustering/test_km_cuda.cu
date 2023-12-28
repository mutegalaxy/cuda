/**
 * @file test_km_cuda.cu
 *
 * @brief Test for the k-means clustering algorithm
 *
 * @author Abas Farah
 * Contact: contact@mutegalaxy.com
 *
 */

#include "km_cuda.h"
#include <iostream>

using namespace std;

int main (int argc, char *argv[]) {
  if (argc != 5) {
    fprintf(stderr, "Wrong input. Need three inputs");
    return 1;
  }

  char *FileName = argv[1];
  int K = atoi(argv[2]);
  int numBlocks = atoi(argv[3]);
  int numThreadsPerBlock = atoi(argv[4]);

  kMeansClustering(FileName, K, numBlocks, numThreadsPerBlock);
 
  return 0;
}
