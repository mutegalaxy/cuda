#define _POSIX_C_SOURCE 199309L
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <vector>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <iomanip>
#include <cmath>

using namespace std;

/* Gives us high-resolution timers. */
/* OSX timer includes */
#ifdef __MACH__
  #include <mach/mach.h>
  #include <mach/mach_time.h>
#endif

/**
* @brief Return the number of seconds since an unspecified time (e.g., Unix
*        epoch). This is accomplished with a high-resolution monotonic timer,
*        suitable for performance timing.
*
* @return The number of seconds.
*/
static inline double monotonic_seconds()
{
#ifdef __MACH__
  /* OSX */
  static mach_timebase_info_data_t info;
  static double seconds_per_unit;
  if(seconds_per_unit == 0) {
    mach_timebase_info(&info);
    seconds_per_unit = (info.numer / info.denom) / 1e9;
  }
  return seconds_per_unit * mach_absolute_time();
#else
  /* Linux systems */
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
#endif
}

__device__ float roundToOneDecimalPlace(float num) {
    return roundf(num * 1.0) / 1.0;
}

/**
* @brief Output the seconds elapsed while clustering.
*
* @param seconds Seconds spent on k-means clustering, excluding IO.
*/
static void print_time(double const seconds)
{
  printf("k-means clustering time: %0.04fs\n", seconds);
}

void printClusters(vector<float> clusters, int K, int D) {
    for (int i = 0; i < K; ++i) {
        cout << "Cluster " << i << ":\n";
        for (int j = 0; j < D; ++j) {
            cout << clusters[i * D + j] << " ";
        }
        cout << "\n";        
    }
}

void printDataPoints(vector<float> dataPoints, int N, int D) {
    for (int i = 0; i < N; ++i) {
    // for (int i = N-1; i < N; ++i) {
        cout << "Point " << i << ":\n";
        for (int j = 0; j < D; ++j) {
            cout << dataPoints[i * D + j] << " ";
        }
        cout << "\n";
    }
    cout << "\n";
}

void printResult(vector<float> result, int N, int K) {
    for (int i = 0; i < N; ++i) {
    // for (int i = 0; i < 1; ++i) {
    // for (int i = N-1; i > N-2; --i) {
    // for (int i = 0; i < N; i+=500000){
        cout << "Point " << i << ":\n";
        for (int j = 0; j < K; ++j) {
            cout << result[i * K + j] << " ";
        }
        cout << "\n";
    }
    cout << "\n";
}

void printMinIndices(vector<int> minIndices, int N) {
    for (int i = 0; i < N; ++i) {
    // for (int i = N-1; i < N; ++i) {
    // for (int i = 0; i < N; i+=500000) {
        cout << "Point " << i << ":\n";
        cout << minIndices[i] << "\n";
    }
    cout << "\n";
}

void printClusterCounts(vector<int> clusterCounts, int K) {
    for (int i = 465; i < K; ++i) {
        cout << "ClusterCount:  " << i << " val: "; 
        cout << clusterCounts[i] << "\n";
    }
    cout << "\n";
}

__global__ void euclideanDistanceKernel(float *dataPoints, float *clusters, float *result, int N, int D, int K) {
    int pointsPerBlock = N / gridDim.x;
    int startN = blockIdx.x * pointsPerBlock;
    int endN = startN + pointsPerBlock;

    int clustersPerThread = K / blockDim.x;
    int startK = threadIdx.x * clustersPerThread;
    int endK = startK + clustersPerThread;

    for (int n = startN; n < endN; ++n) {
        for (int k = startK; k < endK; ++k) {
            float sum = 0.0;
            for (int d = 0; d < D; ++d) {
                float diff = dataPoints[n * D + d] - clusters[k * D + d];
                sum += diff * diff;
            }
            float tmp = sqrtf(sum);
            result[n * K + k] = roundToOneDecimalPlace(tmp);
        }
    }
}

__global__ void reduce(float *distances, int *minIndices, int N, int K){
    int pointsPerBlock = max(N / gridDim.x, 1);

    int startN = blockIdx.x * pointsPerBlock;
    int endN = startN + pointsPerBlock;

    int clustersPerThread = max(K / blockDim.x, 1);

    int startK = threadIdx.x * clustersPerThread;
    int endK = startK + clustersPerThread;

    extern __shared__ int sharedMinIndices[];

    for (int n = startN; n < endN; ++n) {
        float minDistance = FLT_MAX;
        int minIndex = -1;

        for (int k = startK; k < endK; ++k) {
            float dist = distances[n * K + k];
            if (dist < minDistance) {
                minDistance = dist;
                minIndex = k;
            }
        }

        sharedMinIndices[threadIdx.x] = minIndex;

        if (threadIdx.x == 0) {
            for (int i = 1; i < blockDim.x; ++i) {
                if (sharedMinIndices[i] != -1 && distances[n * K + sharedMinIndices[i]] < distances[n * K + minIndex]) {
                    minIndex = sharedMinIndices[i];
                }
            }
            minIndices[n] = minIndex;
        }
    }

}

__global__ void countNumberOfPointsInCluster(int *minIndices, int *clusterCounts, int N, int K) {
    int pointsPerBlock = max(N / gridDim.x, 1);

    int startN = blockIdx.x * pointsPerBlock;
    int endN = startN + pointsPerBlock;

    int clustersPerThread = max(K / blockDim.x, 1);

    int startK = threadIdx.x * clustersPerThread;
    int endK = startK + clustersPerThread;

    for (int n = startN; n < endN; ++n) {
        for (int k = startK; k < endK; ++k) {
            if (minIndices[n] == k) {
                // printf("Point %d is assigned to cluster %d\n", n, k);
                atomicAdd(&clusterCounts[k], 1);
            }
        }
    }
}

__global__ void recomputeCentroidsSum(float *dataPoints, int *minIndices, float *newClusters, int N, int D, int K) {
    int pointsPerBlock = max(N / gridDim.x, 1);

    int startN = blockIdx.x * pointsPerBlock;
    int endN = startN + pointsPerBlock;

    int clustersPerThread = max(K / blockDim.x, 1);

    int startK = threadIdx.x * clustersPerThread;
    int endK = startK + clustersPerThread;

    for (int n = startN; n < endN; ++n) {
        for (int k = startK; k < endK; ++k) {
            if (minIndices[n] == k) {
                for (int d = 0; d < D; ++d) {
                    float sum = dataPoints[n * D + d];
                    // printf("sum: %f\n", sum);
                    atomicAdd(&newClusters[k * D + d], sum);
                    // printf("newClusters[k * D + d]: %f\n", newClusters[k * D + d]);
                }
            }
        }
    }
}

__global__ void divideByClusterCounts(float *newClusters, int *clusterCounts, int K, int D) {
    int clustersPerBlock = max(K / gridDim.x, 1);

    int startK = blockIdx.x * clustersPerBlock;
    int endK = startK + clustersPerBlock;

    for (int k = startK; k < endK; ++k) {
        for (int d = 0; d < D; ++d) {
            if (clusterCounts[k] == 0) {
                continue;
            }
            // printf("currrent cluster value: %f\n", newClusters[k * D + d]);
            // printf("cluster count: %d\n", clusterCounts[k]);
            float tmp = newClusters[k * D + d] / ((float) clusterCounts[k]);
            newClusters[k * D + d] = tmp;
        }
    }
}

__global__ void compareClusters(float *oldClusters, float *newClusters, int *clustersChanged, int K, int D) {
    int clustersPerBlock = max(K / gridDim.x, 1);

    int startK = blockIdx.x * clustersPerBlock;
    int endK = startK + clustersPerBlock;

    int dimensionsPerThread = max(D / blockDim.x, 1);

    int startD = threadIdx.x * dimensionsPerThread;
    int endD = startD + dimensionsPerThread;

    for (int k = startK; k < endK; ++k) {
        for (int d = startD; d < endD; ++d) {
            if (oldClusters[k * D + d] != newClusters[k * D + d]) {
                atomicExch(clustersChanged, 1);
            }
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 5) {
        fprintf(stderr, "Wrong input. Need three inputs");
        return 1;
    }

    char *FileName = argv[1];
    int K = atoi(argv[2]);
    int numBlocks = atoi(argv[3]);
    int numThreadsPerBlock = atoi(argv[4]);
    int N, D;

    ifstream file(FileName);

    file >> N >> D;

    vector<float> dataPoints(N * D);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < D; ++j) {
            float tmp;
            file >> tmp;
            dataPoints[i * D + j] = tmp;
        }
    }

    // printDataPoints(dataPoints, N, D);

    file.close();

    float *d_dataPoints;
    cudaMalloc(&d_dataPoints, N * D * sizeof(float));
    cudaMemcpy(d_dataPoints, dataPoints.data(), N * D * sizeof(float), cudaMemcpyHostToDevice);

    vector<float> clusters(K*D, 0);
    for (int i = 0; i < K; ++i) {
        for(int j = 0; j < D; j+=2){
            // clusters[i*D + j] = (float) i;
            clusters[i*D + j] = dataPoints[i*D + j];
        }
    }
    // printClusters(clusters, K, D);

    float *d_clusters;
    cudaMalloc(&d_clusters, K * D * sizeof(float));
    cudaMemcpy(d_clusters, clusters.data(), K * D * sizeof(float), cudaMemcpyHostToDevice);

    // Init values
    vector<float> pointDistanceFromCluster(N * K);
    vector<int> minClusterIndex(N, INT_MAX);
    int clustersChanged = 0;
    int count = 0;

    float *d_distances;
    int *d_minIndices;
    int *d_clusterCounts;
    float *d_newClusters;
    int *d_clustersChanged;

    cudaMalloc(&d_distances, N * K * sizeof(float));
    cudaMalloc(&d_minIndices, N * sizeof(int));
    cudaMalloc(&d_clusterCounts, K * sizeof(int));
    cudaMalloc(&d_newClusters, K * D * sizeof(float));
    cudaMalloc(&d_clustersChanged, sizeof(int));

    double const start_time = monotonic_seconds();

    // Main loop that does our k means clustering
    while(count < 20){

        // STEP 1:
        dim3 blocks(numBlocks);
        dim3 threads(numThreadsPerBlock);
        euclideanDistanceKernel<<<blocks, threads>>>(d_dataPoints, d_clusters, d_distances, N, D, K);
        cudaMemcpy(pointDistanceFromCluster.data(), d_distances, N * K * sizeof(float), cudaMemcpyDeviceToHost);
        if(count == 1){
            // printResult(pointDistanceFromCluster, N, K);
        }

        cudaMemcpy(d_minIndices, minClusterIndex.data(), N * sizeof(int), cudaMemcpyHostToDevice);

        int sharedMemSize = numThreadsPerBlock * (sizeof(float));
        reduce<<<blocks, threads, sharedMemSize>>>(d_distances, d_minIndices, N, K);

        cudaMemcpy(minClusterIndex.data(), d_minIndices, N * sizeof(int), cudaMemcpyDeviceToHost);
        if(count == 1){
            // printMinIndices(minClusterIndex, N);
        }

        // STEP 2:
        vector<int> clusterCounts(K, 0);
        cudaMemcpy(d_clusterCounts, clusterCounts.data(), K * sizeof(int), cudaMemcpyHostToDevice);

        int numberOfThreadsNeededForCount = max(K / numThreadsPerBlock, 1);
        dim3 threads2(numberOfThreadsNeededForCount);
        countNumberOfPointsInCluster<<<blocks, threads2>>>(d_minIndices, d_clusterCounts, N, K);
        cudaMemcpy(clusterCounts.data(), d_clusterCounts, K * sizeof(int), cudaMemcpyDeviceToHost);
        // if(count == 1){
        //     printClusterCounts(clusterCounts, K);
        // }

        vector<float> newClusters(K * D, 0);
        cudaMemcpy(d_newClusters, newClusters.data(), K * D * sizeof(float), cudaMemcpyHostToDevice);
        recomputeCentroidsSum<<<blocks, threads2>>>(d_dataPoints, d_minIndices, d_newClusters, N, D, K);
        // printClusterCounts(clusterCounts, K);
        // printMinIndices(minClusterIndex, N);
        divideByClusterCounts<<<blocks, threads2>>>(d_newClusters, d_clusterCounts, K, D);
        cudaMemcpy(newClusters.data(), d_newClusters, K * D * sizeof(float), cudaMemcpyDeviceToHost);
        // printClusters(newClusters, K, D);

        clustersChanged = 0;
        cudaMemcpy(d_clustersChanged, &clustersChanged, sizeof(int), cudaMemcpyHostToDevice);
        compareClusters<<<blocks, threads2>>>(d_clusters, d_newClusters, d_clustersChanged, K, D);
        cudaDeviceSynchronize();
        cudaMemcpy(&clustersChanged, d_clustersChanged, sizeof(int), cudaMemcpyDeviceToHost);

        if (clustersChanged == 0) {
            break;
        } 

        // cudaMemcpy(clusters.data(), d_newClusters, K * D * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(d_clusters, d_newClusters, K * D * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(clusters.data(), d_clusters, K * D * sizeof(float), cudaMemcpyDeviceToHost);
        count++;
    }

    double const end_time = monotonic_seconds();
    print_time(end_time - start_time);

    // printMinIndices(clusterCounts, K);

    ofstream clusterFile("clusters.txt");
    clusterFile << fixed << setprecision(3);

    for (int i = 0; i < N; ++i) {
        clusterFile << minClusterIndex[i] << "\n";
    }
    clusterFile.close();

    ofstream centroidFile("centroids.txt");
    centroidFile << fixed << setprecision(3);

    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < D; ++j) {
            centroidFile << clusters[i * D + j] << " ";
        }
        centroidFile << "\n";
    }
    centroidFile.close();

    // printClusters(clusters, K, D);
    // printMinIndices(minClusterIndex, N);
    // cout << "iteration count: " << count << "\n";

    cudaFree(d_dataPoints);
    cudaFree(d_clusters);
    cudaFree(d_distances);
    cudaFree(d_minIndices);
    cudaFree(d_newClusters);
    cudaFree(d_clustersChanged);

    return 0;
}