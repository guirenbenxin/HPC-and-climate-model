CC = /usr/bin/gcc
HIPCC = /public/software/compiler/dtk/dtk-22.04.2/bin/hipcc  ##/public/software/compiler/rocm/rocm-3.9.1/bin/hipcc
CFLAGS =  -fopenmp -g -w -O4 -I..
MPI = -I /opt/hpc/software/mpi/hpcx/v2.7.4/gcc-7.3.1/include
LIB = -L /opt/hpc/software/mpi/hpcx/v2.7.4/gcc-7.3.1/lib -lmpi    

all: 
	${HIPCC} ${CFLAGS} ${MPI} ${LIB} doubleuwshcu.cpp -o uwshcu.exe
#run:
#	./uwshcu.o 4 4 4 4 1 -v -p
clean:
	-rm uwshcu*


