#include <bits/stdc++.h>
#include "../helper/helper.hpp"
using namespace std;
mt19937_64 RNG(chrono::steady_clock::now().time_since_epoch().count());

// reused from assignment 2 code, modified finding the optimal omega part
Result run_sor(const std::vector<double>& A,
               const std::vector<double>& b,
               int n,
               const std::vector<double>& x0,
               const double omega,
               const double tol = 1e-8,
               const int maxit = 1e6)
{
    assert(A.size() == n*n);
    assert(b.size() == n);
    assert(x0.size() == n);

    Result res;
    res.x = x0;

    std::cout << "Chosen Omega = " << omega << '\n';

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
        std::cout << res.iters << ' ' << dx_inf << '\n';
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

int main(){
    // construct the coefficient matrix
    DenseSystem sys = read_system("A.txt", "B.txt");
    cout << "System A matrix is of size (n x n) with n = " << sys.n << endl; 
    print_system(sys);

    // initial guess
    vector<double> x0(sys.n, 0);
    
    // start timer
    auto begin = chrono::high_resolution_clock::now();

    // run the iterative solver change the value of omega here
    Result res = run_sor(sys.A, sys.b, sys.n, x0, 1.17);

    // end timer
    auto end = chrono::high_resolution_clock::now();
    auto elapsed = chrono::duration_cast<chrono::nanoseconds>(end - begin);

    // log data
    cout << "Time elapsed = " << elapsed.count()*(1e-6) << "ms" << endl;
    cout << "Number of iterations = " << res.iters << endl;
}