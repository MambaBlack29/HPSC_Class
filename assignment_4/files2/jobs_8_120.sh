#!/bin/sh
#SBATCH --job-name=mpi_sor120_p8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=00:30:00
#SBATCH --partition=standard
#SBATCH --error=error/job_120_p8.%J.err
#SBATCH --output=output/job_120_p8.%J.out

lscpu
module list

mpirun -np 8 ./sor_120.out
