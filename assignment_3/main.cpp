#include <bits/stdc++.h>
#include "../helper/helper.hpp"
using namespace std;
mt19937_64 RNG(chrono::steady_clock::now().time_since_epoch().count());

// compile with "-fopenmp -O3" flag 

int main(){
    const int num_iters = 1000; // number of times to run the computation
    double total_time = 0.0;
    long long total_solver_iters = 0;
    vector<double> final_x; // store the last solution

    // system from the Kmat and Fvec
    DenseSystem sys = read_system("Kmat.txt", "Fvec.txt");
    cout << "Number of nodes:" << sys.n << endl; 

    // set number of threads
    int num_threads;
    cout << "Enter number of threads (2, 4, 8): " << endl;
    cin >> num_threads;
    math::set_num_threads(num_threads);

    for(int iter = 0; iter < num_iters; ++iter) {
        // initial guess (new random guess for each iteration)
        vector<double> x0(sys.n, 0);
        mt19937 gen(std::random_device{}());
        normal_distribution<double> dist(0.0, 1.0); 
        for(double &val: x0) val = dist(gen);
        
        // start timer
        auto begin = chrono::high_resolution_clock::now();

        // run the iterative solver
        Result res = run_bicgstab(sys.A, sys.b, sys.n, x0);

        // end timer
        auto end = chrono::high_resolution_clock::now();
        auto elapsed = chrono::duration_cast<chrono::nanoseconds>(end - begin);
        
        // accumulate statistics
        total_time += elapsed.count() * (1e-6); // convert to ms
        total_solver_iters += res.iters;

        // store the last solution
        if (iter == num_iters - 1) {
            final_x = res.x;
        }
    }

    // compute averages and log final results
    double avg_time = total_time / num_iters;
    double avg_iters = static_cast<double>(total_solver_iters) / num_iters;

    cout << "Average time elapsed = " << avg_time << "ms" << endl;
    cout << "Average number of iterations = " << avg_iters << endl;
    cout << "Total runs = " << num_iters << endl;

    // save the last solution vector
    save_vector(final_x, "x.txt");
}