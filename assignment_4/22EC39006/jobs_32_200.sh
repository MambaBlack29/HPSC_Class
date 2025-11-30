#!/bin/sh
#SBATCH --job-name=mpi_sor200_p32
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --time=00:30:00
#SBATCH --partition=standard
#SBATCH --error=error/job_200_p32.%J.err
#SBATCH --output=output/job_200_p32.%J.out

lscpu
module list

mpirun -np 32 ./sor_200.out
