#pragma once
#include <vector>
#include <string>
#include <functional>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cmath>
#include "math.hpp"

// Map (i,j) in 0..N-1 x 0..N-1 to row-major index k
inline int idx(int i, int j, int N) { return j * N + i; }

inline std::function<double(double)> const_fn(double val){
    return [val](double){ return val; };
}

static bool read_vector_file(
    const std::string& path, 
    std::vector<double>& out
) {
    std::ifstream in(path);
    if (!in) return false;
    out.clear();
    out.reserve(1<<20);
    double x;
    while (in >> x) out.push_back(x);
    return true;
}

struct DenseSystem {
    std::vector<double> A;
    std::vector<double> b;
    int N = 0;
    int n = 0;
    double h = 0.0;
};

DenseSystem dense_laplace_fdm(
    double h,
    const std::function<double(double)>& g_left,
    const std::function<double(double)>& g_right,
    const std::function<double(double)>& g_bottom,
    const std::function<double(double)>& g_top
) {
    int N = (int)std::llround(1.0 / h - 1.0);
    int n = N * N;

    DenseSystem sys;
    sys.N = N; sys.n = n; sys.h = h;
    sys.A.assign(n * n, 0.0);
    sys.b.assign(n, 0.0);

    for (int j = 0; j < N; ++j) {
        double yj = (j + 1) * h;
        for (int i = 0; i < N; ++i) {
            double xi = (i + 1) * h;

            int r = idx(i, j, N);
            sys.A[r * n + r] = 4.0;

            if (i - 1 >= 0) {
                int c = idx(i - 1, j, N);
                sys.A[r * n + c] = -1.0;
            } else {
                sys.b[r] += g_left(yj);
            }

            if (i + 1 < N) {
                int c = idx(i + 1, j, N);
                sys.A[r * n + c] = -1.0;
            } else {
                sys.b[r] += g_right(yj);
            }

            if (j - 1 >= 0) {
                int c = idx(i, j - 1, N);
                sys.A[r * n + c] = -1.0;
            } else {
                sys.b[r] += g_bottom(xi);
            }

            if (j + 1 < N) {
                int c = idx(i, j + 1, N);
                sys.A[r * n + c] = -1.0;
            } else {
                sys.b[r] += g_top(xi);
            }
        }
    }

    if(!math::symmetric(sys.A, n)){
        sys.A = math::mul(math::add(sys.A, math::transpose(sys.A, n, n)), 0.5);
    }

    return sys;
}

DenseSystem dense_laplace_fdm(
    double h,
    double bc_left, double bc_right,
    double bc_bottom, double bc_top
) {
    auto gl = const_fn(bc_left);
    auto gr = const_fn(bc_right);
    auto gb = const_fn(bc_bottom);
    auto gt = const_fn(bc_top);
    return dense_laplace_fdm(h, gl, gr, gb, gt);
}

DenseSystem read_system(
    const std::string mat_path,
    const std::string vec_path
){
    DenseSystem sys;
    if(!read_vector_file(mat_path, sys.A)) return sys;
    if(!read_vector_file(vec_path, sys.b)) return sys;
    sys.n = sys.b.size();
    sys.N = int(std::sqrt(sys.n));
    return sys;
}

void print_system(const DenseSystem& sys) {
    int n = sys.N * sys.N;
    std::cout.setf(std::ios::fixed); std::cout << std::setprecision(2);
    for (int r = 0; r < n; ++r) {
        for (int c = 0; c < n; ++c) {
            std::cout << std::setw(6) << sys.A[idx(c, r, n)] << " ";
        }
        std::cout << " | " << std::setw(8) << sys.b[r] << "\n";
    }
}

void save_vector(const std::vector<double>& vec, const std::string& filename) {
    std::ofstream out(filename, std::ios::out | std::ios::trunc);
    if (!out.is_open()) return;
    for (double val : vec) out << val << "\n";
    out.close();
}

void save_system(const DenseSystem& sys){
    std::string Kmat = "Kmat" + std::to_string(sys.N) + ".txt";
    std::string Fvec = "Fvec" + std::to_string(sys.N) + ".txt";
    save_vector(sys.A, Kmat);
    save_vector(sys.b, Fvec);
}