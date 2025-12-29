#!/usr/bin/env python3
"""
Analyze speedup from SOR MPI results and generate visualizations.

This script reads the sor_results.csv file and generates:
1. Speedup curves for different grid sizes
2. Strong scaling efficiency plots
3. Detailed analysis table with observations
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
# If the above doesn't work, try:
# plt.style.use('default')

def calculate_speedup(df):
    """
    Calculate speedup for each configuration.
    Speedup = T_baseline / T_parallel
    where T_baseline is the time with 2 processors (baseline)
    
    Efficiency = (Actual_Speedup / Ideal_Speedup) × 100%
    where Ideal_Speedup = P (number of processors used)
    
    For strong scaling from 2-processor baseline:
    - Speedup at P processors = T_2 / T_P
    - Ideal speedup from baseline = P / 2
    - Efficiency = (T_2 / T_P) / (P / 2) × 100%
    - This measures how efficiently we use the ADDITIONAL processors beyond baseline
    """
    speedup_data = []
    
    # Group by grid size
    for grid_size in df['grid_size'].unique():
        grid_data = df[df['grid_size'] == grid_size].sort_values('num_processors')
        
        # Get baseline time (2 processors)
        baseline_time = grid_data[grid_data['num_processors'] == 2]['time_elapsed'].values[0]
        baseline_procs = 2
        
        for _, row in grid_data.iterrows():
            # Speedup relative to 2-processor baseline
            speedup = baseline_time / row['time_elapsed']
            
            # Ideal speedup from baseline (linear scaling)
            ideal_speedup = row['num_processors'] / baseline_procs
            
            # Parallel efficiency: measures how well we utilize processors
            # Efficiency = (Actual Speedup / Ideal Speedup) × 100%
            # Alternative formula: (T_baseline × P_baseline) / (T_current × P_current) × 100%
            efficiency = (baseline_time * baseline_procs) / (row['time_elapsed'] * row['num_processors']) * 100
            
            speedup_data.append({
                'grid_size': row['grid_size'],
                'num_nodes': row['num_nodes'],
                'num_processors': row['num_processors'],
                'time_elapsed': row['time_elapsed'],
                'baseline_time': baseline_time,
                'speedup': speedup,
                'ideal_speedup': ideal_speedup,
                'efficiency': efficiency,
                'iterations': row['iterations'],
                'optimal_omega': row['optimal_omega']
            })
    
    return pd.DataFrame(speedup_data)


def plot_speedup_curves(speedup_df, output_dir):
    """
    Plot speedup curves for different grid sizes.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    markers = ['o', 's', '^']
    
    # Plot for each grid size
    for idx, grid_size in enumerate(sorted(speedup_df['grid_size'].unique())):
        data = speedup_df[speedup_df['grid_size'] == grid_size].sort_values('num_processors')
        num_nodes = data['num_nodes'].iloc[0]
        
        # Actual speedup
        ax.plot(data['num_processors'], data['speedup'], 
                marker=markers[idx], markersize=10, linewidth=2.5,
                label=f'Grid {grid_size}×{grid_size} (N={num_nodes})',
                color=colors[idx])
        
        # Add values on points
        for _, row in data.iterrows():
            ax.annotate(f'{row["speedup"]:.2f}', 
                       (row['num_processors'], row['speedup']),
                       textcoords="offset points", xytext=(0,10), 
                       ha='center', fontsize=9)
    
    # Plot ideal speedup (linear)
    procs = sorted(speedup_df['num_processors'].unique())
    ideal = [p/2 for p in procs]  # normalized to 2 processors
    ax.plot(procs, ideal, 'k--', linewidth=2, label='Ideal (Linear) Speedup', alpha=0.7)
    
    ax.set_xlabel('Number of Processors', fontsize=14, fontweight='bold')
    ax.set_ylabel('Speedup', fontsize=14, fontweight='bold')
    ax.set_title('Strong Scaling: Speedup vs Number of Processors\n(Domain Decomposition MPI SOR Solver)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(procs)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'speedup_curves.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'speedup_curves.png'}")
    
    return fig


def plot_efficiency(speedup_df, output_dir):
    """
    Plot parallel efficiency for different grid sizes.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    markers = ['o', 's', '^']
    
    # Plot for each grid size
    for idx, grid_size in enumerate(sorted(speedup_df['grid_size'].unique())):
        data = speedup_df[speedup_df['grid_size'] == grid_size].sort_values('num_processors')
        num_nodes = data['num_nodes'].iloc[0]
        
        ax.plot(data['num_processors'], data['efficiency'], 
                marker=markers[idx], markersize=10, linewidth=2.5,
                label=f'Grid {grid_size}×{grid_size} (N={num_nodes})',
                color=colors[idx])
        
        # Add values on points
        for _, row in data.iterrows():
            ax.annotate(f'{row["efficiency"]:.1f}%', 
                       (row['num_processors'], row['efficiency']),
                       textcoords="offset points", xytext=(0,10), 
                       ha='center', fontsize=9)
    
    # Plot ideal efficiency (100%)
    procs = sorted(speedup_df['num_processors'].unique())
    ax.axhline(y=100, color='k', linestyle='--', linewidth=2, label='Ideal Efficiency (100%)', alpha=0.7)
    
    ax.set_xlabel('Number of Processors', fontsize=14, fontweight='bold')
    ax.set_ylabel('Parallel Efficiency (%)', fontsize=14, fontweight='bold')
    ax.set_title('Strong Scaling: Parallel Efficiency vs Number of Processors\n(Domain Decomposition MPI SOR Solver)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(procs)
    ax.set_ylim([0, 120])
    
    # plt.tight_layout()
    plt.savefig(output_dir / 'efficiency_curves.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'efficiency_curves.png'}")
    
    return fig


def plot_execution_time(speedup_df, output_dir):
    """
    Plot execution time comparison.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    markers = ['o', 's', '^']
    
    # Plot for each grid size
    for idx, grid_size in enumerate(sorted(speedup_df['grid_size'].unique())):
        data = speedup_df[speedup_df['grid_size'] == grid_size].sort_values('num_processors')
        num_nodes = data['num_nodes'].iloc[0]
        
        ax.plot(data['num_processors'], data['time_elapsed'], 
                marker=markers[idx], markersize=10, linewidth=2.5,
                label=f'Grid {grid_size}×{grid_size} (N={num_nodes})',
                color=colors[idx])
        
        # Add values on points
        for _, row in data.iterrows():
            ax.annotate(f'{row["time_elapsed"]:.2f}s', 
                       (row['num_processors'], row['time_elapsed']),
                       textcoords="offset points", xytext=(0,10), 
                       ha='center', fontsize=9)
    
    ax.set_xlabel('Number of Processors', fontsize=14, fontweight='bold')
    ax.set_ylabel('Execution Time (seconds)', fontsize=14, fontweight='bold')
    ax.set_title('Execution Time vs Number of Processors\n(Domain Decomposition MPI SOR Solver)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(sorted(speedup_df['num_processors'].unique()))
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'execution_time.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'execution_time.png'}")
    
    return fig


def generate_observations(speedup_df, output_dir):
    """
    Generate detailed observations and save to text file.
    """
    observations = []
    observations.append("="*80)
    observations.append("SPEEDUP ANALYSIS AND OBSERVATIONS")
    observations.append("Domain Decomposition MPI SOR Solver with Optimal Omega")
    observations.append("="*80)
    observations.append("")
    
    # Summary table
    observations.append("DETAILED SPEEDUP TABLE:")
    observations.append("-"*80)
    observations.append(f"{'Grid':<12} {'Nodes':<10} {'Procs':<8} {'Time(s)':<12} {'Speedup':<10} {'Efficiency':<12} {'Iterations':<10}")
    observations.append("-"*80)
    
    for _, row in speedup_df.iterrows():
        observations.append(
            f"{row['grid_size']}×{row['grid_size']:<8} "
            f"{row['num_nodes']:<10} "
            f"{row['num_processors']:<8} "
            f"{row['time_elapsed']:<12.3f} "
            f"{row['speedup']:<10.3f} "
            f"{row['efficiency']:<12.2f}% "
            f"{row['iterations']:<10}"
        )
    
    observations.append("-"*80)
    observations.append("")
    
    # Key observations
    observations.append("KEY OBSERVATIONS:")
    observations.append("")
    
    # 1. Overall speedup trends
    observations.append("1. SPEEDUP CHARACTERISTICS:")
    for grid_size in sorted(speedup_df['grid_size'].unique()):
        data = speedup_df[speedup_df['grid_size'] == grid_size].sort_values('num_processors')
        max_speedup = data['speedup'].max()
        max_speedup_procs = data.loc[data['speedup'].idxmax(), 'num_processors']
        observations.append(f"   - Grid {grid_size}×{grid_size} (N={data['num_nodes'].iloc[0]}):")
        observations.append(f"     Maximum speedup: {max_speedup:.2f}x at {max_speedup_procs} processors")
        observations.append(f"     Speedup ratio: 2p→8p: {data[data['num_processors']==8]['speedup'].values[0]/data[data['num_processors']==2]['speedup'].values[0]:.2f}x")
        observations.append(f"                    8p→16p: {data[data['num_processors']==16]['speedup'].values[0]/data[data['num_processors']==8]['speedup'].values[0]:.2f}x")
        observations.append(f"                    16p→32p: {data[data['num_processors']==32]['speedup'].values[0]/data[data['num_processors']==16]['speedup'].values[0]:.2f}x")
        observations.append("")
    
    # 2. Efficiency analysis
    observations.append("2. PARALLEL EFFICIENCY:")
    for grid_size in sorted(speedup_df['grid_size'].unique()):
        data = speedup_df[speedup_df['grid_size'] == grid_size].sort_values('num_processors')
        observations.append(f"   - Grid {grid_size}×{grid_size}:")
        for _, row in data.iterrows():
            observations.append(f"     {row['num_processors']} processors: {row['efficiency']:.2f}% efficiency")
        observations.append("")
    
    # 3. Scaling behavior
    observations.append("3. SCALING BEHAVIOR:")
    observations.append("   - All grid sizes show sub-linear speedup (below ideal linear scaling)")
    observations.append("   - Larger problem sizes (200×200) maintain better efficiency at higher processor counts")
    observations.append("   - Diminishing returns observed as processor count increases due to:")
    observations.append("     * Increased communication overhead between MPI processes")
    observations.append("     * Domain decomposition leading to smaller subdomain per processor")
    observations.append("     * Synchronization costs (MPI_Barrier, MPI_Allreduce)")
    observations.append("")
    
    # 4. Problem size dependency
    observations.append("4. PROBLEM SIZE DEPENDENCY:")
    eff_32_small = speedup_df[(speedup_df['grid_size']==80) & (speedup_df['num_processors']==32)]['efficiency'].values[0]
    eff_32_large = speedup_df[(speedup_df['grid_size']==200) & (speedup_df['num_processors']==32)]['efficiency'].values[0]
    observations.append(f"   - At 32 processors:")
    observations.append(f"     Grid 80×80:   {eff_32_small:.2f}% efficiency")
    observations.append(f"     Grid 200×200: {eff_32_large:.2f}% efficiency")
    observations.append(f"   - Larger grids maintain better efficiency (more work per communication)")
    observations.append("")
    
    # 5. Optimal omega impact
    observations.append("5. OPTIMAL OMEGA PARAMETER:")
    for grid_size in sorted(speedup_df['grid_size'].unique()):
        omega = speedup_df[speedup_df['grid_size']==grid_size]['optimal_omega'].iloc[0]
        observations.append(f"   - Grid {grid_size}×{grid_size}: ω = {omega:.5f}")
    observations.append("   - Optimal omega increases with grid size (approaching 2.0 for large grids)")
    observations.append("   - Higher omega → faster convergence → fewer iterations")
    observations.append("")
    
    # 6. Iteration count analysis
    observations.append("6. CONVERGENCE ITERATIONS:")
    for grid_size in sorted(speedup_df['grid_size'].unique()):
        data = speedup_df[speedup_df['grid_size'] == grid_size].sort_values('num_processors')
        observations.append(f"   - Grid {grid_size}×{grid_size}:")
        for _, row in data.iterrows():
            observations.append(f"     {row['num_processors']} processors: {row['iterations']} iterations")
        observations.append("")
    
    # 7. Recommendations
    observations.append("7. RECOMMENDATIONS:")
    observations.append("   - For small problems (80×80): Use 8-16 processors for good efficiency")
    observations.append("   - For medium problems (120×120): Use 16 processors for optimal balance")
    observations.append("   - For large problems (200×200): Can scale up to 32 processors efficiently")
    observations.append("   - Beyond these points, communication overhead dominates computation")
    observations.append("")
    
    observations.append("="*80)
    
    # Write to file
    obs_file = output_dir / 'observations.txt'
    with open(obs_file, 'w') as f:
        f.write('\n'.join(observations))
    
    print(f"Saved: {obs_file}")
    
    # Also print to console
    print("\n" + '\n'.join(observations))


def main():
    """Main function to analyze speedup and generate visualizations."""
    
    # Get script directory
    script_dir = Path(__file__).parent
    csv_file = script_dir / 'sor_results.csv'
    
    if not csv_file.exists():
        print(f"Error: {csv_file} not found!")
        print("Please run parse_results.py first to generate the CSV file.")
        return
    
    # Read CSV data
    print(f"Reading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    # Calculate speedup
    print("Calculating speedup metrics...")
    speedup_df = calculate_speedup(df)
    
    # Save speedup data to CSV
    speedup_csv = script_dir / 'speedup_analysis.csv'
    speedup_df.to_csv(speedup_csv, index=False)
    print(f"Saved: {speedup_csv}")
    
    # Generate plots
    print("\nGenerating visualizations...")
    plot_speedup_curves(speedup_df, script_dir)
    plot_efficiency(speedup_df, script_dir)
    plot_execution_time(speedup_df, script_dir)
    
    # Generate observations
    print("\nGenerating observations...")
    generate_observations(speedup_df, script_dir)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print(f"  1. {script_dir / 'speedup_curves.png'} - Speedup vs Processors")
    print(f"  2. {script_dir / 'efficiency_curves.png'} - Efficiency vs Processors")
    print(f"  3. {script_dir / 'execution_time.png'} - Execution Time vs Processors")
    print(f"  4. {script_dir / 'speedup_analysis.csv'} - Detailed speedup data")
    print(f"  5. {script_dir / 'observations.txt'} - Detailed analysis and observations")
    print("="*80)


if __name__ == '__main__':
    main()
