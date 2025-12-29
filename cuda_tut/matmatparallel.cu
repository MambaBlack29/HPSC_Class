#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
int n=1000;

__global__ void paramul(int *ad, int *bd, int *cd,int n)
{
  int k;/*printf ("hi");*/
  int idx=blockIdx.x*blockDim.x+threadIdx.x;
  int idy=blockIdx.y*blockDim.y+threadIdx.y;
  //printf ("%d", idx);
  if(idx<n && idy<n)
  {
    for(k=0;k<n;k++)
    (*(cd+n*idx+idy))=(*(cd+n*idx+idy))+(*(ad+n*idx+k))*(*(bd+n*k+idy));
  }
__syncthreads();
}

void matmul(int*ah,int*bh,int*ch)
{
  int*ad;int*bd;int*cd;
  int size=n*n*sizeof(int);
  cudaMalloc((void**)&ad,size);
  cudaMalloc((void**)&bd,size);
  cudaMalloc((void**)&cd,size);
  cudaMemcpy(ad,ah,size,cudaMemcpyHostToDevice);
  cudaMemcpy(bd,bh,size,cudaMemcpyHostToDevice);
  int g;
  int d=16;
  g=ceil(n/d);
  dim3 dimGrid (g,g);
  dim3 dimBlock (d,d);
//  printf("ho");

  paramul <<<dimGrid,dimBlock>>>(ad,bd,cd,n);

  cudaMemcpy(ch,cd,size,cudaMemcpyDeviceToHost);
  cudaFree(ad);
  cudaFree(bd);
  cudaFree(cd);

}

int main ()
{
  double total_time=0.000;
  	clock_t begin=clock();
  int *ah;int*bh;int*ch;int i,j;
  ah=(int*)malloc(n*n*sizeof(int));
  bh=(int*)malloc(n*n*sizeof(int));
  ch=(int*)malloc(n*n*sizeof(int));
  for(i=0;i<n;i++)
  {
    for(j=0;j<n;j++)
    {
      *(ah+n*i+j)=i+1;
      *(bh+n*i+j)=1;
      *(ch+n*i+j)=0;
    }
  }
matmul(ah,bh,ch);
/*for(i=0;i<n;i++)
{
  for(j=0;j<n;j++)
  { printf ("%d ",(*(ch+n*i+j)));
  }
  printf("\n");
}
*/

free(ah);
free(bh);
free(ch);
clock_t end =clock();
total_time += ((double) (end - begin)) / CLOCKS_PER_SEC;
printf(" time %lf\n",total_time*1000 );

}
