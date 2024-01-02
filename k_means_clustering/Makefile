NVCC_FLAGS = --compiler-options -ansi --compiler-options -Wall --compiler-options -std=c++17

all: test_km_cuda

test_km_cuda: test_km_cuda.o libkm_cuda.a
	nvcc test_km_cuda.o libkm_cuda.a -o test_km_cuda
	rm -f *.o *.a

test_km_cuda.o: test_km_cuda.cu
	nvcc $(NVCC_FLAGS) -c test_km_cuda.cu -o test_km_cuda.o

libkm_cuda.a: km_cuda.o
	ar rcs libkm_cuda.a km_cuda.o

km_cuda.o: km_cuda.cu
	nvcc $(NVCC_FLAGS) -c km_cuda.cu -o km_cuda.o

.PHONY: clean

clean:
	rm -f *.o *.a test_km_cuda clusters.txt centroids.txt
