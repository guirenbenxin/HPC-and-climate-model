#include "hip/hip_runtime.h"
#include <iostream>
#include <stdio.h>
#include <hip/hip_runtime.h>
#include <math.h>
#include <time.h>
//#include "omp.h"
//#include "cuda/common.h"
//#include <gptl.h>
///#include <string>

using namespace std;



__global__ void compute_uwshcu()
{


}

// ! ------------------------ !
// !                          ! 
// ! End of subroutine blocks !
// !                          !
// ! ------------------------ !



void compute_uwshcu_inv_( )
{

		hipLaunchKernelGGL(compute_uwshcu, dim3(ceil((1024)/128.0)), dim3(128), 0, 0, 

			);
  //hipDeviceSynchronize();
}

int main(int argc, char *argv[])
{
	
	compute_uwshcu_inv_( );

	

	return 0;
}



