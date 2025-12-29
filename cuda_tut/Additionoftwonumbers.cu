/******************************************************************************************
Saiyedul Islam
saiyedul.islam@gmail.com

******************
System Configuration
******************
Processor:		                    Intel(R) Pentium(R) Dual  CPU  E2200  @ 2.20GHz (2201 MHz)
Operating System:	                Windows 7 Ultimate, 32-bit
DirectX version:                    11.0 
GPU processor:		                GeForce 9400 GT
Driver version:	                    257.21
CUDA Cores:		                    16 
Core clock:		                    550 MHz 
Shader clock:		                1350 MHz
Memory clock:	                	400 MHz (800 MHz data rate) 
Memory interface:	                128-bit 
Total available graphics memory:	1791 MB
Dedicated video memory:	            1024 MB DDR2
System video memory:	            0 MB
Shared system memory:	            767 MB
Video BIOS version:	                62.94.61.00.58
IRQ:	                    		16
Bus:			                    PCI Express x16

*************
Device Query
*************
Device 0: "GeForce 9400 GT"
  CUDA Driver Version:                           	3.0
  CUDA Runtime Version:                          	2.30
  CUDA Capability Major revision number:    	    1
  CUDA Capability Minor revision number:         	1
  Total amount of global memory:                 	1073414144 bytes
  Number of multiprocessors:                     	2
  Number of cores:                               	16
  Total amount of constant memory:           		65536 bytes
  Total amount of shared memory per block:	        16384 bytes
  Total number of registers available per block: 	8192
  Warp size:                                     	32
  Maximum number of threads per block:   	        512
  Maximum sizes of each dimension of a block:	    512 x 512 x 64
  Maximum sizes of each dimension of a grid:   	    65535 x 65535 x 1
  Maximum memory pitch:                         	262144 bytes
  Texture alignment:                             	256 bytes
  Clock rate:                                    	1.35 GHz
  Concurrent copy and execution: 		            Yes
  Run time limit on kernels:                     	Yes
  Integrated:                                    	No
  Support host page-locked memory mapping:	        No
  Compute mode:                                  	Default (multiple host threads can use this device simultaneously)

*****************
Problem Statement
*****************
Program to Add two numbers on GPU.

****************************************************************************************/

#include<stdio.h>
//#include<conio.h>

__global__ void add_number(float *ad,float *bd)
{

        *ad += *bd;                             //adding values in GPU memory
}
int main()
{
	float *a,*b;
	float *ad,*bd;
	size_t size = sizeof(float);

    //allocate memory on host
	a=(float*)malloc(size);
	b=(float*)malloc(size);
	
    //allocate memory on device
	cudaMalloc(&ad,size);
	//printf("\nAfter cudaMalloc for ad\n%s\n",cudaGetErrorString(cudaGetLastError()));
	cudaMalloc(&bd,size);
	//printf("\nAfter cudaMalloc for bd\n%s\n",cudaGetErrorString(cudaGetLastError()));
	
    printf("\nEnter two numbers\n");
    scanf("%f%f",a,b);

	//copy data from host memory to device memory
	cudaMemcpy(ad,a,size,cudaMemcpyHostToDevice);
    //printf("\nAfter HostToDevice Memcpy for ad\n%s\n",cudaGetErrorString(cudaGetLastError()));
	cudaMemcpy(bd,b,size,cudaMemcpyHostToDevice);
    //printf("\nAfter HostToDevice Memcpy for bd\n%s\n",cudaGetErrorString(cudaGetLastError()));
	
    //GPU timer code
    float time;
    cudaEvent_t start,stop;			
	cudaEventCreate(&start);
    cudaEventCreate(&stop);
	cudaEventRecord(start,0);
	
    //launch kernel with only one thread
    add_number<<< 1,1 >>>(ad,bd);
	
    cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time,start,stop);			//time taken in kernel call calculated
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

    //copy back the results
	cudaMemcpy(a,ad,size,cudaMemcpyDeviceToHost);
	//printf("\nAfter DeviceToHost Memcpy for a\n%s\n",cudaGetErrorString(cudaGetLastError()));
	
	//print the results
	printf("\nAddition of above two numbers on GPU evaluates to = %f",*a);
    printf("\n\nTime taken is %f (ms)\n",time);
    
    //deallocate host and device memories
    cudaFree(ad); cudaFree(bd);
	free(a);free(b);

//	_getch();
    return 1;
}
