#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
int n=30000;
void dotp (int *d, int r[n],int k[n], int n)
{
  int i,j;
  for(i=0;i<n;i++)
  for(j=0;j<n;j++)
  k[i]=k[i]+(d[n*i+j]*r[j]);
//printf("this is k in func %lf\n",k);


}

void main()
{
  int n=30000;int i,j;
  int *a; int b[n];int k[n];int c;
  a=(int*)malloc(n*n*sizeof(int));

  for( i=0;i<n;i++)
  {
  b[i]=abs(rand()%7+3);
  k[i]=0;
  for(j=0;j<n;j++)
  {a[n*i+j]=abs(rand()%4+1);}

//  printf("%d  %d\n",a[i],b[i]);
  }
 double total_time=0.000;
  	clock_t begin=clock();
dotp(a,b,k,n);
//printf("%d\n",c );
clock_t end =clock();
total_time += ((double) (end - begin)) / CLOCKS_PER_SEC;
printf(" time %lf\n",total_time );
}
