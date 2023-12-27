all: km_cuda

km_cuda: km_cuda.cu
	nvcc -o km_cuda --compiler-options -ansi --compiler-options -Wall --compiler-options -std=c++17 km_cuda.cu

clean:
	rm km_cuda