#include <stdio.h>
#include <stdlib.h>
#include <time.h>
int n=1000;
int matmul(int a[n][n], int b[n][n],int c[n][n], int n)
{ int sum=0;int i,j;int k=0;
  for(int i=0;i<n;++i)
  {sum=0;
    for(int j=0;j<n;++j)
    {
      for(int h=0;h<n;h++)
      {
        sum=sum+a[i][h]*b[h][j];
      }
      c[i][j]=sum;
    }

  }
}
void main()
{ double total_time=0.000;
	clock_t begin=clock();
  int i,j;

  int a[n][n]; int b[n][n]; int c[n][n];
  for(int i=0;i<n;i++)
  {
    for(int j=0;j<n;j++)
  {
    a[i][j]=i+1;
    b[i][j]=i+1;
    c[i][j]=0;
  }
}
matmul(a,b,c,n);
/*for(i=0;i<n;i++)
{
  for( j=0;j<n;j++)
{

printf("%d " ,c[i][j]);
}printf("\n");
}*//*free (a);
free(b);
free(c);*/
clock_t end =clock();
total_time += ((double) (end - begin)) / CLOCKS_PER_SEC;
printf(" time %lf\n",total_time*1000 );
}
