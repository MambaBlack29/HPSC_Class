#include <bits/stdc++.h>
#include "../helper/helper.hpp"
using namespace std;

pair<double, long long> run_once(const DenseSystem &sys, Result &res){
    // initial guess
    vector<double> x0(sys.n, 0);
    mt19937 gen(std::random_device{}());
    normal_distribution<double> dist(0.0, 1.0); 
    for(double &val: x0) val = dist(gen);
    
    // start timer
    auto begin = chrono::high_resolution_clock::now();

    // run the iterative solver (CHANGE SOLVER HERE)
    res = run_jacobi(sys.A, sys.b, sys.n, x0);

    // end timer
    auto end = chrono::high_resolution_clock::now();
    auto elapsed = chrono::duration_cast<chrono::nanoseconds>(end - begin);

    return {elapsed.count()*(1e-6), res.iters};
}

int main(){
    // boundary conditions and size
    double h = 0.08;
    double bc_left = 0, bc_right = 0, bc_bottom = 0, bc_top = 1;

    // construct the coefficient matrix
    DenseSystem sys = read_system("Kmat.txt", "Fvec.txt");
    Result res;
    // save_system(sys);

    double time = 0, iters = 0;
    int max_count = 100; // CHANGE TIMES RUN HERE
    for(int i = 0; i < max_count; i++){
        auto out = run_once(sys, res);
        time += out.first;
        iters += out.second;
    }
    time /= max_count;
    iters /= max_count;

    // log data
    cout << "Time elapsed = " << time << "ms" << endl;
    cout << "Number of iterations = " << iters << endl;

    save_vector(res.x, "x.txt");
}