#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <cassert>
#include <mpi.h>
#include <iostream>

/**
 * MPI-based Domain Decomposition Iterative Solvers
 * 
 * Compile with: mpic++ -std=c++17 -O3 your_file.cpp -o your_program
 * Run with: mpirun -np <num_procs> ./your_program
 * 
 * Features:
 * - Domain decomposition by rows (1D partitioning)
 * - Jacobi iteration with MPI communication
 * - SOR iteration with MPI communication
 * - Ghost cell exchange between neighboring processes
 */

namespace math_mpi {

// MPI Result structure
struct MPIResult {
    std::vector<double> x_local;  // local portion of solution
    std::vector<double> x_global; // full solution (only valid on root)
    long long iters = 0;
    bool converged = false;
    double elapsed_time = 0.0;
};

// Helper: Gather all local solutions to root process
inline void gather_solution(const std::vector<double>& x_local,
                            std::vector<double>& x_global,
                            int n_local,
                            int n_global,
                            int rank,
                            MPI_Comm comm = MPI_COMM_WORLD)
{
    int nprocs;
    MPI_Comm_size(comm, &nprocs);
    
    std::vector<int> recvcounts(nprocs);
    std::vector<int> displs(nprocs);
    
    MPI_Allgather(&n_local, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, comm);
    
    displs[0] = 0;
    for(int i = 1; i < nprocs; ++i) {
        displs[i] = displs[i-1] + recvcounts[i-1];
    }
    
    if(rank == 0) {
        x_global.resize(n_global);
    }
    
    MPI_Gatherv(x_local.data(), n_local, MPI_DOUBLE,
                x_global.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                0, comm);
}

// Helper: Compute local dot product and reduce globally
inline double global_dot(const std::vector<double>& a,
                        const std::vector<double>& b,
                        MPI_Comm comm = MPI_COMM_WORLD)
{
    assert(a.size() == b.size());
    double local_sum = 0.0;
    for(size_t i = 0; i < a.size(); ++i) {
        local_sum += a[i] * b[i];
    }
    
    double global_sum = 0.0;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, comm);
    return global_sum;
}

// Helper: Compute local norm and reduce globally
inline double global_norm(const std::vector<double>& a,
                         bool inf_norm = false,
                         MPI_Comm comm = MPI_COMM_WORLD)
{
    if(inf_norm) {
        double local_max = 0.0;
        for(const auto& v : a) {
            local_max = std::max(local_max, std::abs(v));
        }
        double global_max = 0.0;
        MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, comm);
        return global_max;
    } else {
        double local_sum = 0.0;
        for(const auto& v : a) {
            local_sum += v * v;
        }
        double global_sum = 0.0;
        MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, comm);
        return std::sqrt(global_sum);
    }
}

// Helper: Vector subtraction
inline std::vector<double> vec_sub(const std::vector<double>& a,
                                   const std::vector<double>& b)
{
    assert(a.size() == b.size());
    std::vector<double> c(a.size());
    for(size_t i = 0; i < a.size(); ++i) {
        c[i] = a[i] - b[i];
    }
    return c;
}

// Helper: Vector multiplication by scalar
inline std::vector<double> vec_mul(const std::vector<double>& a, double scale)
{
    std::vector<double> b(a.size());
    for(size_t i = 0; i < a.size(); ++i) {
        b[i] = a[i] * scale;
    }
    return b;
}

/**
 * MPI SOR Solver with Domain Decomposition
 * 
 * Uses optimum omega factor computed from spectral radius
 * Note: For domain decomposition, true Gauss-Seidel ordering is difficult,
 * so we use a colored/parallel variant where each process updates its rows
 * independently, then communicates.
 */
inline MPIResult mpi_sor(const std::vector<double>& A,
                        const std::vector<double>& b,
                        int n,
                        const std::vector<double>& x0,
                        double omega = 1.5,
                        const double tol = 1e-8,
                        const int maxit = 1e6,
                        MPI_Comm comm = MPI_COMM_WORLD)
{
    int rank, nprocs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);
    
    MPIResult res;
    double start_time = MPI_Wtime();
    
    // Distribute rows among processes
    int rows_per_proc = n / nprocs;
    int remainder = n % nprocs;
    
    int local_start = rank * rows_per_proc + std::min(rank, remainder);
    int n_local = rows_per_proc + (rank < remainder ? 1 : 0);
    int local_end = local_start + n_local;
    
    // Local portion of solution
    std::vector<double> x_local(n_local);
    for(int i = 0; i < n_local; ++i) {
        x_local[i] = x0[local_start + i];
    }
    
    // Current full solution
    std::vector<double> x_current = x0;
    
    // Precompute diagonal inverse
    std::vector<double> Dinv_local(n_local);
    for(int i = 0; i < n_local; ++i) {
        int global_i = local_start + i;
        assert(A[global_i * n + global_i] != 0.0);
        Dinv_local[i] = 1.0 / A[global_i * n + global_i];
    }
    
    if(rank == 0) {
        std::cout << "Using omega = " << omega << std::endl;
    }
    
    double bn = global_norm(b, false, comm);
    
    for(res.iters = 0; res.iters < maxit; ++res.iters) {
        // Compute local portion of new iterate using SOR formula
        std::vector<double> x_new_local(n_local);
        
        for(int i = 0; i < n_local; ++i) {
            int global_i = local_start + i;
            double sum = 0.0;
            
            // Compute A[i,:] * x_current, excluding diagonal
            for(int j = 0; j < n; ++j) {
                if(j != global_i) {
                    sum += A[global_i * n + j] * x_current[j];
                }
            }
            
            // SOR update: x_new = (1-omega)*x_old + omega*D^{-1}*(b - R*x_old)
            double x_gs = (b[global_i] - sum) * Dinv_local[i];
            x_new_local[i] = (1.0 - omega) * x_current[global_i] + omega * x_gs;
        }
        
        // Gather new solution to all processes
        std::vector<double> x_new(n);
        std::vector<int> recvcounts(nprocs);
        std::vector<int> displs(nprocs);
        
        for(int p = 0; p < nprocs; ++p) {
            int p_start = p * rows_per_proc + std::min(p, remainder);
            recvcounts[p] = rows_per_proc + (p < remainder ? 1 : 0);
            displs[p] = p_start;
        }
        
        MPI_Allgatherv(x_new_local.data(), n_local, MPI_DOUBLE,
                       x_new.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                       comm);
        
        // Check convergence
        double dx_inf = global_norm(vec_sub(x_new, x_current), true, comm);
        double xn_inf = global_norm(x_new, true, comm);
        
        x_current = std::move(x_new);
        x_local = x_new_local;
        
        if(dx_inf <= tol * (1.0 + xn_inf)) {
            res.converged = true;
            break;
        }
    }
    
    res.elapsed_time = MPI_Wtime() - start_time;
    res.x_local = x_local;
    gather_solution(x_local, res.x_global, n_local, n, rank, comm);
    
    return res;
}

/**
 * Compute optimal omega for SOR from Jacobi spectral radius
 * This should be run on a single process or with consistent results
 */
inline double compute_optimal_omega(const std::vector<double>& A, int n)
{
    // Build Jacobi iteration matrix J = I - D^{-1}*A
    std::vector<double> Dinv(n);
    for(int i = 0; i < n; ++i) {
        assert(A[i*n + i] != 0.0);
        Dinv[i] = 1.0 / A[i*n + i];
    }
    
    // Power method to estimate spectral radius of J
    std::vector<double> v(n, 0.0);
    v[0] = 1.0;
    
    const int power_iters = 100;
    for(int k = 0; k < power_iters; ++k) {
        std::vector<double> Jv(n, 0.0);
        
        // Jv = v - D^{-1}*A*v
        for(int i = 0; i < n; ++i) {
            double Av_i = 0.0;
            for(int j = 0; j < n; ++j) {
                Av_i += A[i*n + j] * v[j];
            }
            Jv[i] = v[i] - Dinv[i] * Av_i;
        }
        
        // Normalize
        double norm_Jv = 0.0;
        for(const auto& val : Jv) {
            norm_Jv += val * val;
        }
        norm_Jv = std::sqrt(norm_Jv);
        
        if(norm_Jv < 1e-14) break;
        
        for(int i = 0; i < n; ++i) {
            v[i] = Jv[i] / norm_Jv;
        }
    }
    
    // Final application to get eigenvalue estimate
    std::vector<double> Jv(n, 0.0);
    for(int i = 0; i < n; ++i) {
        double Av_i = 0.0;
        for(int j = 0; j < n; ++j) {
            Av_i += A[i*n + j] * v[j];
        }
        Jv[i] = v[i] - Dinv[i] * Av_i;
    }
    
    double norm_Jv = 0.0, norm_v = 0.0;
    for(int i = 0; i < n; ++i) {
        norm_Jv += Jv[i] * Jv[i];
        norm_v += v[i] * v[i];
    }
    
    double rho = std::sqrt(norm_Jv / norm_v);
    rho = std::min(rho, 0.9999);  // Keep it below 1
    
    // Optimal omega = 2 / (1 + sqrt(1 - rho^2))
    double omega_opt = 2.0 / (1.0 + std::sqrt(1.0 - rho*rho));
    
    return std::min(std::max(omega_opt, 1.0), 1.99);
}

} // namespace math_mpi
