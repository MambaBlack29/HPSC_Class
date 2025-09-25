#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <cassert>
#include "math.hpp"

// Fixed-point iteration result
struct Result {
    std::vector<double> x; // answer
    long long iters = 0; // number of iterations
    bool converged = false; // reached convergence or not
};

// gauss seidel
Result run_gauss_seidel(const std::vector<double>& A,
                        const std::vector<double>& b,
                        int n,
                        const std::vector<double>& x0,
                        const double tol = 1e-8,
                        const int maxit = 1e6)
{
    assert(A.size() == static_cast<size_t>(n)*n);
    assert(b.size() == static_cast<size_t>(n));
    assert(x0.size() == static_cast<size_t>(n));

    Result res;
    res.x = x0;

    // check diagonal and (optionally) build 1/diag
    std::vector<double> Dinv(n, 0.0);
    for(int i = 0; i < n; ++i){
        assert(A[i*n + i] != 0.0);
        Dinv[i] = 1.0 / A[i*n + i];
    }

    double bn = math::norm(b);
    for(res.iters = 0; res.iters < maxit; ++res.iters){
        std::vector<double> xnew = res.x; // will be updated in-place (lower part uses xnew)
        for(int i = 0; i < n; ++i){
            double sum = 0.0;
            // lower part: use updated values from xnew
            for(int j = 0; j < i; ++j) sum += A[i*n + j] * xnew[j];
            // upper part: use old values from res.x
            for(int j = i+1; j < n; ++j) sum += A[i*n + j] * res.x[j];

            // Gauss-Seidel update (omega = 1)
            xnew[i] = (b[i] - sum) * Dinv[i];
        }

        double dx_inf = math::norm(math::sub(xnew, res.x), true);
        double xn_inf = math::norm(xnew, true);

        res.x = std::move(xnew);
        if(dx_inf <= tol * (1.0 + xn_inf)){
            res.converged = true;
            return res;
        }
    }

    res.converged = false;
    return res;
}

// Jacobi
Result run_jacobi(const std::vector<double>& A,
                  const std::vector<double>& b,
                  int n,
                  const std::vector<double>& x0,
                  const double tol = 1e-8,
                  const int maxit = 1e6)
{
    assert(A.size() == static_cast<size_t>(n)*n);
    assert(b.size() == static_cast<size_t>(n));
    assert(x0.size() == static_cast<size_t>(n));

    Result res;
    res.x = x0;

    // check diagonal and precompute 1/diag
    std::vector<double> Dinv(n, 0.0);
    for(int i = 0; i < n; ++i){
        assert(A[i*n + i] != 0.0);
        Dinv[i] = 1.0 / A[i*n + i];
    }

    double bn = math::norm(b);
    for(res.iters = 0; res.iters < maxit; ++res.iters){
        // compute xnew entirely from old res.x
        std::vector<double> xnew(n, 0.0);

        for(int i = 0; i < n; ++i){
            double sum = 0.0;
            for(int j = 0; j < n; ++j){
                if(j == i) continue;
                sum += A[i*n + j] * res.x[j];
            }
            xnew[i] = (b[i] - sum) * Dinv[i];
        }

        double dx_inf = math::norm(math::sub(xnew, res.x), true);
        double xn_inf = math::norm(xnew, true);

        res.x = std::move(xnew);
        if(dx_inf <= tol * (1.0 + xn_inf)){
            res.converged = true;
            return res;
        }
    }

    res.converged = false;
    return res;
}

// successive over relaxation
Result run_sor_opt(const std::vector<double>& A,
                   const std::vector<double>& b,
                   int n,
                   const std::vector<double>& x0,
                   const double tol = 1e-8,
                   const int maxit = 1e6)
{
    assert(A.size() == n*n);
    assert(b.size() == n);
    assert(x0.size() == n);

    Result res;
    res.x = x0;

    // ---- Build D^{-1} and check diagonal ----
    std::vector<double> Dinv(n, 0.0);
    for(int i = 0; i < n; ++i){
        assert(A[i*n + i] > 0);
        Dinv[i] = 1.0 / A[i*n + i];
    }

    // ---- Power method to estimate rho(J), J = I - D^{-1}A ----
    // We avoid lambdas; just write steps inline.
    double rho = 0.0;
    {
        std::vector<double> v(n, 0.0), Av, Jv;
        v[0] = 1.0;

        const int pm_iters = 50;
        for(int k = 0; k < pm_iters; ++k){
            Av = math::matmul(A, v, n, n);
            Jv.assign(n, 0.0); 
            Jv = math::sub(v, math::mul(Dinv, Av));
            double jn = math::norm(Jv);
            if(jn == 0.0){ rho = 0.0; break; }
            v = math::mul(Jv, 1.0/jn);
        }
        // one last application to estimate scale
        Av = math::matmul(A, v, n, n);
        Jv.assign(n, 0.0);
        Jv = math::sub(v, math::mul(Dinv, Av));
        double jn = math::norm(Jv);
        double vn = math::norm(v);
        rho = (vn > 0.0) ? (jn / vn) : 0.0;
        if(!(rho >= 0.0) || std::isnan(rho) || std::isinf(rho)) rho = 0.0;
        rho = std::min(rho, 0.999999); // keep inside [0,1)
    }

    // ---- Optimal omega: 2 / (1 + sqrt(1 - rho^2)) ----
    double omega = 2.0 / (1 + std::sqrt(1 - rho*rho));
    double eps = 1e-8;
    if(omega <= eps) omega = eps;
    else if(omega >= 2.0) omega = 2.0 - eps;
    std::cout << "Optimal Omega = " << omega << '\n';

    // ---- Iteration (Gauss–Seidel sweep with relaxation) ----
    // dx_inf <= tol * (1 + ||x_new||_inf)
    for(res.iters = 0; res.iters < maxit; ++res.iters){
        std::vector<double> xnew = res.x;

        for(int i = 0; i < n; ++i){
            double sum = 0.0;

            // lower part: uses updated xnew[j]
            for(int j = 0; j < i;   ++j) sum += A[i*n + j] * xnew[j];
            // upper part: uses old res.x[j]
            for(int j = i+1; j < n; ++j) sum += A[i*n + j] * res.x[j];

            xnew[i] = (1.0-omega) * res.x[i] + (omega/A[i*n+i]) * (b[i]-sum);
        }

        double dx_inf = math::norm(math::sub(xnew, res.x), true);
        double xn_inf = math::norm(xnew, true);

        res.x = std::move(xnew);
        if(dx_inf <= tol * (1.0 + xn_inf)){
            res.converged = true;
            return res;
        }
    }

    res.converged = false;
    return res;
}

// steepest descent
Result run_stp_desc(const std::vector<double>& A,
                    const std::vector<double>& b,
                    int n,
                    const std::vector<double>& x0,
                    const double tol = 1e-8,
                    const int maxit = 1e6)
{
    assert(A.size() == n*n);
    assert(b.size() == n);
    assert(x0.size() == n);

    Result res;
    res.x = x0;
    std::vector<double> rk = math::sub(b, math::matmul(A, x0, n, n));
    double bn = math::norm(b);
    
    for(res.iters = 0; res.iters < maxit; res.iters++){
        // compute p_k and alpha_k
        std::vector<double> pk = math::matmul(A, rk, n, n);
        double alphak = math::dot(rk, rk) / math::dot(rk, pk);
        if(std::isinf(alphak) || alphak < 0) break;

        // update x_k and r_k
        res.x = math::add(res.x, math::mul(rk, alphak));
        rk = math::sub(rk, math::mul(pk, alphak));

        // break if norm(rk) small
        double rn = math::norm(rk);
        if(rn <= tol * bn){
            res.converged = true;
            return res;
        }
    }
    res.converged = false;
    return res;
}

// minimum residual
Result run_min_res(const std::vector<double>& A,
                   const std::vector<double>& b,
                   int n,
                   const std::vector<double>& x0,
                   const double tol = 1e-8,
                   const int maxit = 1e6)
{
    assert(A.size() == n*n);
    assert(b.size() == n);
    assert(x0.size() == n);

    Result res;
    res.x = x0;
    std::vector<double> rk = math::sub(b, math::matmul(A, x0, n, n));
    double bn = math::norm(b);
    
    for(res.iters = 0; res.iters < maxit; res.iters++){
        // compute p_k and alpha_k
        std::vector<double> pk = math::matmul(A, rk, n, n);
        double alphak = math::dot(pk, rk) / math::dot(pk, pk);
        if(std::isinf(alphak) || alphak < 0) break;

        // update x_k and r_k
        res.x = math::add(res.x, math::mul(rk, alphak));
        rk = math::sub(rk, math::mul(pk, alphak));

        // break if norm(rk) small
        double rn = math::norm(rk);
        if(rn <= tol * bn){
            res.converged = true;
            return res;
        }
    }
    res.converged = false;
    return res;
}

// conjudate gradient
Result run_conj_grad(const std::vector<double>& A,
                     const std::vector<double>& b,
                     int n,
                     const std::vector<double>& x0,
                     const double tol = 1e-8,
                     const int maxit = 1e6)
{
    assert(A.size() == n*n);
    assert(b.size() == n);
    assert(x0.size() == n);

    Result res;
    res.x = x0;
    std::vector<double> rk = math::sub(b, math::matmul(A, x0, n, n));
    std::vector<double> pk = rk;
    double bn = math::norm(b);
    
    for(res.iters = 0; res.iters < maxit; res.iters++){
        // compute A*p_k and alpha_k
        std::vector<double> Apk = math::matmul(A, pk, n, n);
        double alphak = math::dot(rk, rk) / math::dot(Apk, pk);
        if(std::isinf(alphak) || alphak < 0) break;

        // update x_k and r_k
        res.x = math::add(res.x, math::mul(pk, alphak));
        std::vector<double> next_rk = math::sub(rk, math::mul(Apk, alphak));

        // compute beta_k and update p_k and r_k
        double betak = math::dot(next_rk, next_rk) / math::dot(rk, rk);
        pk = math::add(next_rk, math::mul(pk, betak));
        rk = std::move(next_rk);

        // break if norm(rk) small
        double rn = math::norm(rk);
        if(rn <= tol * bn){
            res.converged = true;
            return res;
        }
    }
    res.converged = false;
    return res;
}

// Biconjugate gradient stabilised
Result run_bicgstab(const std::vector<double>& A,
                    const std::vector<double>& b,
                    int n,
                    const std::vector<double>& x0,
                    const double tol = 1e-8,
                    const int maxit = 1e6)
{
    assert(A.size() == n*n);
    assert(b.size() == n);
    assert(x0.size() == n);

    Result res;
    res.x = x0;
    std::vector<double> rk = math::sub(b, math::matmul(A, x0, n, n));
    std::vector<double> r0 = rk;
    std::vector<double> pk = rk;
    double bn = math::norm(b);
    
    for(res.iters = 0; res.iters < maxit; res.iters++){
        // compute A*p_k, alpha_k and s_k
        std::vector<double> Apk = math::matmul(A, pk, n, n);
        double alphak = math::dot(rk, r0) / math::dot(Apk, r0);
        if(std::isinf(alphak) || alphak < 0) break;
        std::vector<double> sk = math::sub(rk, math::mul(Apk, alphak));

        // compute A*s_k and omega_k
        std::vector<double> Ask = math::matmul(A, sk, n, n);
        double omegak = math::dot(Ask, sk) / math::dot(Ask, Ask);

        // update x_k and r_k
        res.x = math::add(res.x, math::mul(pk, alphak));
        res.x = math::add(res.x, math::mul(sk, omegak));
        std::vector<double> next_rk = math::sub(sk, math::mul(Ask, omegak));

        // compute beta_k and update p_k and r_k
        double betak = math::dot(next_rk, r0) / math::dot(rk, r0);
        betak *= (alphak / omegak);
        pk = math::add(next_rk, math::mul(pk, betak));
        pk = math::sub(pk, math::mul(Apk, betak*omegak));
        rk = std::move(next_rk);

        // break if norm(rk) small
        double rn = math::norm(rk);
        if(rn <= tol * bn){
            res.converged = true;
            return res;
        }
    }
    res.converged = false;
    return res;
}