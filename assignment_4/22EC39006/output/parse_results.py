#!/usr/bin/env python3
"""
Parse MPI SOR output files and compile results into a CSV file.

This script extracts the following information from each output file:
- Grid size (inferred from filename)
- Number of nodes (grid_size^2)
- Number of cores/processors
- Optimal omega value
- Number of iterations until convergence
- Time elapsed until convergence
- Error at convergence
"""

import os
import re
import csv
from pathlib import Path


def parse_output_file(filepath):
    """
    Parse a single output file and extract relevant metrics.
    
    Args:
        filepath: Path to the output file
        
    Returns:
        Dictionary containing parsed data, or None if parsing fails
    """
    filename = os.path.basename(filepath)
    
    # Extract grid size and number of processors from filename
    # Format: job_{grid_size}_p{num_processors}.{job_id}.out
    match = re.match(r'job_(\d+)_p(\d+)\.(\d+)\.out', filename)
    if not match:
        print(f"Warning: Could not parse filename: {filename}")
        return None
    
    grid_size = int(match.group(1))
    num_processors = int(match.group(2))
    num_nodes = grid_size * grid_size
    
    # Initialize variables
    optimal_omega = None
    iterations = None
    error = None
    time_elapsed = None
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            
            # Extract optimal omega
            omega_match = re.search(r'Computed optimal omega:\s+([\d.]+)', content)
            if omega_match:
                optimal_omega = float(omega_match.group(1))
            
            # Extract iteration number and error from the convergence line
            # Format: "Iteration #1739: \t Error: 8.881784e-16"
            iter_match = re.search(r'Iteration #(\d+):\s+Error:\s+([\d.e+-]+)', content)
            if iter_match:
                iterations = int(iter_match.group(1))
                error = float(iter_match.group(2))
            
            # Extract time elapsed
            # Format: "Time elapsed for the calculation: 49.087395 seconds"
            time_match = re.search(r'Time elapsed for the calculation:\s+([\d.]+)\s+seconds', content)
            if time_match:
                time_elapsed = float(time_match.group(1))
            
            # Alternative: Extract omega from "Using SOR with optimal omega" line
            if optimal_omega is None:
                omega_match2 = re.search(r'Using SOR with optimal omega\s*=\s*([\d.]+)', content)
                if omega_match2:
                    optimal_omega = float(omega_match2.group(1))
    
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return None
    
    # Validate that we extracted all required data
    if None in [optimal_omega, iterations, error, time_elapsed]:
        print(f"Warning: Incomplete data for {filename}")
        print(f"  omega={optimal_omega}, iter={iterations}, error={error}, time={time_elapsed}")
    
    return {
        'filename': filename,
        'grid_size': grid_size,
        'num_nodes': num_nodes,
        'num_processors': num_processors,
        'optimal_omega': optimal_omega,
        'iterations': iterations,
        'error': error,
        'time_elapsed': time_elapsed
    }


def main():
    """Main function to parse all output files and generate CSV."""
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    output_dir = script_dir
    
    # Find all output files
    output_files = sorted(output_dir.glob('job_*.out'))
    
    if not output_files:
        print(f"No output files found in {output_dir}")
        return
    
    print(f"Found {len(output_files)} output files")
    
    # Parse all files
    results = []
    for filepath in output_files:
        print(f"Parsing {filepath.name}...")
        data = parse_output_file(filepath)
        if data:
            results.append(data)
    
    # Sort results by grid_size, then by num_processors
    results.sort(key=lambda x: (x['grid_size'], x['num_processors']))
    
    # Write to CSV
    csv_path = output_dir / 'sor_results.csv'
    
    fieldnames = [
        'filename',
        'grid_size',
        'num_nodes',
        'num_processors',
        'optimal_omega',
        'iterations',
        'error',
        'time_elapsed'
    ]
    
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nSuccessfully created CSV file: {csv_path}")
    print(f"Total records: {len(results)}")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Filename':<30} {'Grid':<8} {'Nodes':<8} {'Procs':<6} {'Omega':<8} {'Iters':<8} {'Time(s)':<10}")
    print("-"*80)
    for r in results:
        print(f"{r['filename']:<30} {r['grid_size']:<8} {r['num_nodes']:<8} "
              f"{r['num_processors']:<6} {r['optimal_omega']:<8.4f} "
              f"{r['iterations']:<8} {r['time_elapsed']:<10.3f}")


if __name__ == '__main__':
    main()
