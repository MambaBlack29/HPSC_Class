#!/bin/sh
#SBATCH --job-name=mpi_sor200_p2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --time=10:30:00
#SBATCH --partition=standard
#SBATCH --error=error/job_200_p2.%J.err
#SBATCH --output=output/job_200_p2.%J.out

lscpu
module list

mpirun -np 2 ./sor_200.out
