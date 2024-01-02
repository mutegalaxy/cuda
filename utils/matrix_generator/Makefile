NVCC_FLAGS = --compiler-options -ansi --compiler-options -Wall --compiler-options -std=c++17

all: matrix_generator

test: test_matrix_generator

test_matrix_generator: test_matrix_generator.o libmatrix_generator.a
	nvcc test_matrix_generator.o libmatrix_generator.a -o test_matrix_generator
	rm -f *.o *.a

test_matrix_generator.o: test_matrix_generator.cu
	nvcc $(NVCC_FLAGS) -c test_matrix_generator.cu -o test_matrix_generator.o

libmatrix_generator.a: matrix_generator.o
	ar rcs libmatrix_generator.a matrix_generator.o

matrix_generator: matrix_generator.cu
	nvcc $(NVCC_FLAGS) -o matrix_generator matrix_generator.cu

matrix_generator.o: matrix_generator.cu
	nvcc $(NVCC_FLAGS) -c matrix_generator.cu -o matrix_generator.o

BIN_DIR = ../../bin
ABS_BIN_DIR = $(shell cd $(BIN_DIR); pwd)

.PHONY: install
install: matrix_generator
	@mkdir -p $(BIN_DIR)
	@cp matrix_generator $(BIN_DIR)
	@echo "matrix_generator installed in $(BIN_DIR)"
	@echo "Add '$(ABS_BIN_DIR)' to your PATH to use matrix_generator"

.PHONY: clean
clean:
	rm -f *.o *.a matrix_generator test_matrix_generator
