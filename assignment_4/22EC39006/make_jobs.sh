#!/bin/bash

mkdir -p output error

# jobs_2_80.sh
cat > jobs_2_80.sh << 'EOF'
#!/bin/sh
#SBATCH --job-name=mpi_sor80_p2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --time=00:30:00
#SBATCH --partition=standard
#SBATCH --error=error/job_80_p2.%J.err
#SBATCH --output=output/job_80_p2.%J.out

lscpu
module list

mpirun -np 2 ./sor_80.out
EOF

# jobs_8_80.sh
cat > jobs_8_80.sh << 'EOF'
#!/bin/sh
#SBATCH --job-name=mpi_sor80_p8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=00:30:00
#SBATCH --partition=standard
#SBATCH --error=error/job_80_p8.%J.err
#SBATCH --output=output/job_80_p8.%J.out

lscpu
module list

mpirun -np 8 ./sor_80.out
EOF

# jobs_16_80.sh
cat > jobs_16_80.sh << 'EOF'
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
EOF

# jobs_32_80.sh
cat > jobs_32_80.sh << 'EOF'
#!/bin/sh
#SBATCH --job-name=mpi_sor80_p32
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --time=00:30:00
#SBATCH --partition=standard
#SBATCH --error=error/job_80_p32.%J.err
#SBATCH --output=output/job_80_p32.%J.out

lscpu
module list

mpirun -np 32 ./sor_80.out
EOF

# jobs_2_120.sh
cat > jobs_2_120.sh << 'EOF'
#!/bin/sh
#SBATCH --job-name=mpi_sor120_p2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --time=00:30:00
#SBATCH --partition=standard
#SBATCH --error=error/job_120_p2.%J.err
#SBATCH --output=output/job_120_p2.%J.out

lscpu
module list

mpirun -np 2 ./sor_120.out
EOF

# jobs_8_120.sh
cat > jobs_8_120.sh << 'EOF'
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
EOF

# jobs_16_120.sh
cat > jobs_16_120.sh << 'EOF'
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
EOF

# jobs_32_120.sh
cat > jobs_32_120.sh << 'EOF'
#!/bin/sh
#SBATCH --job-name=mpi_sor120_p32
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --time=00:30:00
#SBATCH --partition=standard
#SBATCH --error=error/job_120_p32.%J.err
#SBATCH --output=output/job_120_p32.%J.out

lscpu
module list

mpirun -np 32 ./sor_120.out
EOF

# jobs_2_200.sh
cat > jobs_2_200.sh << 'EOF'
#!/bin/sh
#SBATCH --job-name=mpi_sor200_p2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --time=00:30:00
#SBATCH --partition=standard
#SBATCH --error=error/job_200_p2.%J.err
#SBATCH --output=output/job_200_p2.%J.out

lscpu
module list

mpirun -np 2 ./sor_200.out
EOF

# jobs_8_200.sh
cat > jobs_8_200.sh << 'EOF'
#!/bin/sh
#SBATCH --job-name=mpi_sor200_p8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=00:30:00
#SBATCH --partition=standard
#SBATCH --error=error/job_200_p8.%J.err
#SBATCH --output=output/job_200_p8.%J.out

lscpu
module list

mpirun -np 8 ./sor_200.out
EOF

# jobs_16_200.sh
cat > jobs_16_200.sh << 'EOF'
#!/bin/sh
#SBATCH --job-name=mpi_sor200_p16
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=00:30:00
#SBATCH --partition=standard
#SBATCH --error=error/job_200_p16.%J.err
#SBATCH --output=output/job_200_p16.%J.out

lscpu
module list

mpirun -np 16 ./sor_200.out
EOF

# jobs_32_200.sh
cat > jobs_32_200.sh << 'EOF'
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
EOF

chmod 644 jobs_*_*.sh

