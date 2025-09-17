#include <bits/stdc++.h>
#include "../helper/helper.hpp"
using namespace std;
mt19937_64 RNG(chrono::steady_clock::now().time_since_epoch().count());

int main(){
    // boundary conditions and size
    double h = 0.08;
    double bc_left = 0, bc_right = 0, bc_bottom = 0, bc_top = 1;

    // construct the coefficient matrix
    DenseSystem sys = build_dense_laplace(h, bc_left, bc_right, 
                                          bc_bottom, bc_top);
    // save_system(sys);

    // initial guess
    vector<double> x0(sys.n, 0);
    // mt19937 gen(std::random_device{}());
    // normal_distribution<double> dist(0.0, 1.0); 
    // for(double &val: x0) val = dist(gen);
    
    // start timer
    auto begin = chrono::high_resolution_clock::now();

    // run the iterative solver
    Result res = run_bicgstab(sys.A, sys.b, sys.n, x0);

    // end timer
    auto end = chrono::high_resolution_clock::now();
    auto elapsed = chrono::duration_cast<chrono::nanoseconds>(end - begin);

    // log data
    cout << "Time elapsed = " << elapsed.count()*(1e-6) << "ms" << endl;
    cout << "Number of iterations = " << res.iters << endl;

    save_vector(res.x, "x.txt");
}