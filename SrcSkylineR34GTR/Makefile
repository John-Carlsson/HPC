#tests
all : rebuild test
all_d : rebuild_d test

#builds
rebuild : clean setup source
rebuild_d : clean setup source_debug

setup : 
	mkdir Intermediate

test :
	./source > results.csv
	echo "saved results in results.csv"

#main app
source : math_cpu math_cuda math_tensor math_cublas
	g++ -o source -fopenmp -L/usr/local/cuda-11.8/lib64 ./Src/source.cpp ./Intermediate/math_cpu.o ./Intermediate/math_cuda.o ./Intermediate/math_tensor.o ./Intermediate/math_cublas.o -lcuda -lcudart -O3 -mavx512f -lcublas

source_debug : math_cpu_debug math_cuda_debug math_tensor_debug math_cublas_debug
	g++ -g -o source -fopenmp -L/usr/local/cuda-11.8/lib64 ./Src/source.cpp ./Intermediate/math_cpu.o ./Intermediate/math_cuda.o ./Intermediate/math_tensor.o ./Intermediate/math_cublas.o -lcuda -lcudart -lcublas


#cublas implememtation
math_cublas:
	nvcc -c -o ./Intermediate/math_cublas.o -arch=sm_75 ./Src/math_cublas.cu -O3

math_cublas_debug:
	nvcc -g -c -o ./Intermediate/math_cublas.o -arch=sm_75 ./Src/math_cublas.cu

#tensor implememtation
math_tensor: math_cuda
	nvcc -c -o ./Intermediate/math_tensor.o -arch=sm_75 ./Src/math_tensor.cu -O3

math_tensor_debug: math_cuda_debug
	nvcc -g -c -o ./Intermediate/math_tensor.o -arch=sm_75 ./Src/math_tensor.cu


#cuda implememtation
math_cuda:
	nvcc -c -o ./Intermediate/math_cuda.o -arch=sm_75 ./Src/math_cuda.cu -O3

math_cuda_debug:
	nvcc -g -c -o ./Intermediate/math_cuda.o -arch=sm_75 ./Src/math_cuda.cu


#cpu implememtation
math_cpu:
	g++ -c -o ./Intermediate/math_cpu.o ./Src/math_cpu.cpp -fopenmp -O3 -mavx512f

math_cpu_debug:
	g++ -g -c -o ./Intermediate/math_cpu.o ./Src/math_cpu.cpp -fopenmp


clean:
	-rm -f ./Intermediate/*.o ./Intermediate/*.gch source
	rmdir Intermediate
