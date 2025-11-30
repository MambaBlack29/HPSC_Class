#!/bin/bash

# sor_80 runs
sbatch jobs_2_80.sh
sbatch jobs_8_80.sh
sbatch jobs_16_80.sh
sbatch jobs_32_80.sh

# sor_120 runs
sbatch jobs_2_120.sh
sbatch jobs_8_120.sh
sbatch jobs_16_120.sh
sbatch jobs_32_120.sh

# sor_200 runs
sbatch jobs_2_200.sh
sbatch jobs_8_200.sh
sbatch jobs_16_200.sh
sbatch jobs_32_200.sh

