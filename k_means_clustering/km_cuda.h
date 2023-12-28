/**
 * @defgroup k_meands_clustering km_cuda
 *
 * @brief Parrallel implementation of k-means clustering using CUDA.
 *
 * @author Abas Farah
 * Contact: contact@mutegalaxy.com
 *
 */

#ifndef KM_CUDA_H
#define KM_CUDA_H

#include <vector>

using namespace std;

/* kMeansClustering: Runs the k-means clustering algorithm on GPU using CUDA.
 *                   Given a file of datapoints it will cluster them into K clusters.
 *                   It will use CUDA to parallelize the algorithm.
 *                   The number of blocks and threads per block needs to be specified.
 *                   The clusters and centroids will be written to a file.
 *                       - clusters.txt
 *                       - centroids.txt
*/ 
void kMeansClustering(char* FileName, int K, int numBlocks, int numThreadsPerBlock);

#endif /* KM_CUDA_H */
