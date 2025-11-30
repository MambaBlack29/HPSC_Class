#!/home/mambablack29/miniconda3/envs/ml/bin/python
"""
Speedup Analysis for MPI Domain Decomposition Solvers

Reads benchmark results and generates speedup curves and analysis

Prerequisites: conda activate ml
               pip install pandas matplotlib numpy
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid X11 errors
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def calculate_speedup(df, method, size):
    """Calculate speedup relative to smallest number of processes"""
    method_size_data = df[(df['method'] == method) & (df['matrix_size'] == size)]
    
    if method_size_data.empty:
        return None
    
    method_size_data = method_size_data.sort_values('num_procs')
    
    # Use 2 processes as baseline (or smallest available)
    baseline_procs = method_size_data['num_procs'].min()
    baseline_time = method_size_data[method_size_data['num_procs'] == baseline_procs]['time'].values[0]
    
    speedup_data = []
    for _, row in method_size_data.iterrows():
        speedup = baseline_time / row['time']
        efficiency = speedup / (row['num_procs'] / baseline_procs)
        speedup_data.append({
            'num_procs': row['num_procs'],
            'time': row['time'],
            'speedup': speedup,
            'efficiency': efficiency,
            'iterations': row['iterations']
        })
    
    return pd.DataFrame(speedup_data)

def plot_speedup_curves(df, output_file='speedup_curves.png'):
    """Generate speedup curves for different matrix sizes"""
    
    sizes = sorted(df['matrix_size'].unique())
    
    # Single plot for SOR only
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    for size in sizes:
        speedup_df = calculate_speedup(df, 'SOR', size)
        
        if speedup_df is not None and not speedup_df.empty:
            # Normalize speedup to be relative to 2 processes = 1.0
            ax.plot(speedup_df['num_procs'], speedup_df['speedup'], 
                   marker='o', linewidth=2, markersize=8,
                   label=f'n={size}')
    
    # Add ideal speedup line
    procs = sorted(df['num_procs'].unique())
    if len(procs) > 0:
        baseline_procs = procs[0]
        ideal_speedup = [p / baseline_procs for p in procs]
        ax.plot(procs, ideal_speedup, 'k--', alpha=0.5, linewidth=1.5, label='Ideal')
    
    ax.set_xlabel('Number of Processes', fontsize=12)
    ax.set_ylabel('Speedup', fontsize=12)
    ax.set_title('SOR Method with Optimal Omega', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    ax.set_xticks(procs)
    ax.set_xticklabels([str(p) for p in procs])
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Speedup curves saved to: {output_file}")
    plt.close()

def plot_efficiency(df, output_file='efficiency_curves.png'):
    """Generate efficiency curves"""
    
    sizes = sorted(df['matrix_size'].unique())
    
    # Single plot for SOR only
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    for size in sizes:
        speedup_df = calculate_speedup(df, 'SOR', size)
        
        if speedup_df is not None and not speedup_df.empty:
            ax.plot(speedup_df['num_procs'], speedup_df['efficiency'] * 100, 
                   marker='s', linewidth=2, markersize=8,
                   label=f'n={size}')
    
    # Add 100% efficiency line
    procs = sorted(df['num_procs'].unique())
    ax.axhline(y=100, color='k', linestyle='--', alpha=0.5, linewidth=1.5, label='Ideal (100%)')
    
    ax.set_xlabel('Number of Processes', fontsize=12)
    ax.set_ylabel('Parallel Efficiency (%)', fontsize=12)
    ax.set_title('SOR Method with Optimal Omega', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    ax.set_xticks(procs)
    ax.set_xticklabels([str(p) for p in procs])
    ax.set_ylim([0, 110])
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Efficiency curves saved to: {output_file}")
    plt.close()

def generate_summary_table(df, output_file='speedup_summary.txt'):
    """Generate a summary table of results"""
    
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("MPI DOMAIN DECOMPOSITION BENCHMARK RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        for method in sorted(df['method'].unique()):
            f.write(f"\n{method} METHOD\n")
            f.write("-" * 80 + "\n")
            
            for size in sorted(df['matrix_size'].unique()):
                f.write(f"\nMatrix Size: {size} x {size}\n")
                f.write(f"{'Procs':<10} {'Time(s)':<15} {'Iters':<10} {'Speedup':<15} {'Efficiency(%)':<15}\n")
                f.write("-" * 80 + "\n")
                
                speedup_df = calculate_speedup(df, method, size)
                
                if speedup_df is not None and not speedup_df.empty:
                    for _, row in speedup_df.iterrows():
                        f.write(f"{row['num_procs']:<10} "
                               f"{row['time']:<15.4f} "
                               f"{int(row['iterations']):<10} "
                               f"{row['speedup']:<15.3f} "
                               f"{row['efficiency']*100:<15.2f}\n")
                
                f.write("\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("OBSERVATIONS\n")
        f.write("=" * 80 + "\n")
        f.write("""
1. SOR Method with Optimal Omega:
   - Uses optimal relaxation factor computed from Jacobi spectral radius
   - Formula: ω_opt = 2 / (1 + sqrt(1 - ρ²))
   - Accelerates convergence compared to basic iterative methods

2. Speedup Trends:
   - As the number of processes increases, we observe speedup in computation time
   - The speedup is sub-linear due to communication overhead and synchronization
   - Larger matrices generally achieve better speedup as computation dominates communication

3. Efficiency Analysis:
   - Parallel efficiency decreases as we add more processes
   - This is expected due to increased communication-to-computation ratio
   - SOR shows fast convergence with optimal omega parameter

4. Scalability:
   - Strong scaling study shows how well the code performs with fixed problem size
   - Communication overhead becomes more significant with more processes
   - Domain decomposition introduces synchronization points (Allgatherv) after each iteration
   - MPI_Allgatherv is needed to exchange updated solution components

5. Communication Pattern:
   - All-to-all communication via MPI_Allgatherv after each iteration
   - Global reductions for convergence checking (MPI_Allreduce)
   - Synchronization barriers between benchmark runs

6. Recommended Configuration:
   - For smaller problems (n~6400), using 2-8 processes provides good efficiency
   - For larger problems (n~40000), 16-32 processes can be effective
   - Beyond a certain point, communication overhead dominates gains from parallelization
""")
    
    print(f"Summary table saved to: {output_file}")

def main():
    print("MPI Speedup Analysis")
    print("===================")
    print("Note: Using non-interactive matplotlib backend to avoid X11 errors")
    print()
    
    # Check if results file exists
    if not os.path.exists('mpi_benchmark_results.csv'):
        print("Error: mpi_benchmark_results.csv not found!")
        print("Please run the benchmark first: ./run_mpi_benchmark.sh")
        sys.exit(1)
    
    # Read results
    print("Reading benchmark results...")
    df = pd.read_csv('mpi_benchmark_results.csv')
    
    if df.empty:
        print("Error: No data in results file!")
        sys.exit(1)
    
    print(f"Found {len(df)} benchmark results")
    print(f"Matrix sizes: {sorted(df['matrix_size'].unique())}")
    print(f"Process counts: {sorted(df['num_procs'].unique())}")
    print(f"Method: SOR with Optimal Omega")
    print()
    
    # Generate visualizations
    print("Generating speedup curves...")
    plot_speedup_curves(df)
    
    print("Generating efficiency curves...")
    plot_efficiency(df)
    
    print("Generating summary table...")
    generate_summary_table(df)
    
    print("\nAnalysis complete!")
    print("\nGenerated files:")
    print("  - speedup_curves.png (SOR speedup vs. number of processes)")
    print("  - efficiency_curves.png (SOR parallel efficiency)")
    print("  - speedup_summary.txt (Detailed numerical results and observations)")

if __name__ == '__main__':
    main()
