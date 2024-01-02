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

#include <vector>

/**
* @brief matrix_generator generates a N x M matrix of random integers.
*
* @param N Number of rows in matrix.
* @param M Number of columns in matrix.
*/
std::vector<int> matrix_generator(int N, int M);

#endif // MATRIX_GENERATOR_H
