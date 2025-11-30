#!/bin/sh
#SBATCH --job-name=mpi_sor80_p16
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=00:30:00
#SBATCH --partition=standard
#SBATCH --error=error/job_80_p16.%J.err
#SBATCH --output=output/job_80_p16.%J.out

lscpu
module list

mpirun -np 16 ./sor_80.out
