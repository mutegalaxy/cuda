/**
 * @defgroup utils/matrix_generator matrix_generator
 *
 * @brief Matrix generator utility.
 *
 * @author Abas Farah
 * Contact: contact@mutegalaxy.com
 *
 */
#ifndef MATRIX_GENERATOR_H
#define MATRIX_GENERATOR_H

#include <cuda_runtime.h>
#include <curand_kernel.h>

/* CUDA kernel function
 *
 * fillMatrixKernal: Fills a matrix with random numbers.
 *
 * @param matrix The matrix to fill.
 * @param N The number of rows.
 * @param M The number of columns.
 *
 * @return void
*/
__global__ void fillMatrixKernal(int *matrix, int N, int M);

#endif // MATRIX_GENERATOR_H
