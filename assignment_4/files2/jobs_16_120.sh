#!/bin/sh
#SBATCH --job-name=mpi_sor120_p16
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=00:30:00
#SBATCH --partition=standard
#SBATCH --error=error/job_120_p16.%J.err
#SBATCH --output=output/job_120_p16.%J.out

lscpu
module list

mpirun -np 16 ./sor_120.out
