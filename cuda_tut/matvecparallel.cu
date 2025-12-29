
#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <time.h>
//int n=2560; int u=16;
__global__ void mvm (int *a, int *x, int *b, int n,int u)
{
  int j=blockIdx.y * blockDim.y + threadIdx.y;
  int i=blockIdx.x * blockDim.x + threadIdx.x;
  int ind= i+gridDim.x*u*j;
  if(ind<n)
  {
    int l;int m=(ind*n);
    *(b+ind)=0;
    for(l=0;l<n;l++)
    *(b+ind)=(*(b+ind))+(*(a+m+l))*(*(x+l));
  }
__syncthreads();
}

void launchmvm(int *ah, int*xh, int *ch, int n, int u)
{
  int*ad;int*xd;int*cd;
  int size=n*sizeof(int);
  int size2=n*n*sizeof(int);
  cudaMalloc((void**)&ad,size2);
  cudaMalloc((void**)&xd,size);
  cudaMalloc((void**)&cd,size);
  cudaMemcpy(ad,ah,size2,cudaMemcpyHostToDevice);
  cudaMemcpy(xd,xh,size,cudaMemcpyHostToDevice);

  int d=ceil(n/256);
  dim3 dimGrid(1,d);
  dim3 dimBlock(16, 16);

  mvm<<<dimGrid,dimBlock>>>(ad,xd,cd,n,u);


  cudaMemcpy(ch,cd,size,cudaMemcpyDeviceToHost);
  cudaFree(ad);
  cudaFree(xd);
  cudaFree(cd);
}
int main ()
{

  int n=30000; int u=16;
  int *ah;
  ah=(int*)malloc(n*n*sizeof(int));
   int xh[n]; int ch[n];
  for(int i=0;i<n;i++)
  {
    xh[i]=rand()%20+1;
    ch[i]=0;
    for(int j=0;j<n;j++)
    {
    ah[n*i+j]=rand()%9+1;
    }
  }
  float time;
  double total_time=0.000;
  clock_t begin=clock();
launchmvm(ah,xh,ch,n,u);
clock_t end =clock();
total_time += ((double) (end - begin)) / CLOCKS_PER_SEC;
printf("\nTime taken is %f \n",total_time);


}
