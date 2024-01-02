all: matrix_generator

matrix_generator: matrix_generator.cu
	nvcc --std=c++17 -o matrix_generator matrix_generator.cu

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
	rm matrix_generator
