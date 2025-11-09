#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <cassert>
#include <omp.h> 

/**
 * add "math::set_num_threads(num_threads)" in code
 * compile with "-fopenmp -O3" flag 
 */

namespace math {

// ---------- Thread control (always-OMP) ----------
inline int& _omp_nthreads_ref() {
    static int n = omp_get_max_threads();
    return n;
}
inline void set_num_threads(int n_threads) { _omp_nthreads_ref() = (n_threads > 0 ? n_threads : 1); }
inline int  get_num_threads() { return _omp_nthreads_ref(); }

// ---------- CONSTANT MATRICES ----------
template<typename T>
inline std::vector<T> identity(int n){
    std::vector<T> A(n*n, T(0));
    #pragma omp parallel for schedule(static) num_threads(math::get_num_threads())
    for(int i = 0; i < n; ++i) A[i*n + i] = T(1);
    return A;
}

template<typename T>
inline std::vector<T> ones(int n, T val = T(1)){
    std::vector<T> A(n*n);
    #pragma omp parallel for schedule(static) num_threads(math::get_num_threads())
    for (int i = 0; i < n*n; ++i) A[i] = val;
    return A;
}

template <typename T>
inline std::vector<T> diag(const std::vector<T>& A, int n, bool vec = false){
    assert(A.size() == size_t(n)*size_t(n));
    std::vector<T> B;
    if(vec) B.assign(n, T(0));
    else    B.assign(n*n, T(0));
    #pragma omp parallel for schedule(static) num_threads(math::get_num_threads())
    for(int i = 0; i < n; i++){
        if(vec) B[i] = A[i*n + i];
        else    B[i*n + i] = A[i*n + i];
    }
    return B;
}

template <typename T>
inline std::vector<T> upper(const std::vector<T>& A, int n, int offset = 1){
    assert(A.size() == size_t(n)*size_t(n));
    std::vector<T> B(n*n, T(0));
    #pragma omp parallel for schedule(static) num_threads(math::get_num_threads())
    for(int i = 0; i < n; ++i){
        for(int j = i+offset; j < n; ++j){
            B[i*n + j] = A[i*n + j];
        }
    }
    return B;
}

template <typename T>
inline std::vector<T> lower(const std::vector<T>& A, int n, int offset = 1){
    assert(A.size() == size_t(n)*size_t(n));
    std::vector<T> B(n*n, T(0));
    #pragma omp parallel for schedule(static) num_threads(math::get_num_threads())
    for(int i = 0; i < n; ++i){
        for(int j = 0; j <= i-offset; ++j){
            if (j >= 0)
                B[i*n + j] = A[i*n + j];
        }
    }
    return B;
}

// ---------- MATRIX OPERATIONS ----------
template <typename T>
inline std::vector<T> matmul(const std::vector<T>& A, const std::vector<T>& B, 
                             int m, int k, int n){
    assert(A.size() == size_t(m)*size_t(k));
    assert(B.size() == size_t(k)*size_t(n));
    std::vector<T> C(m*n, T(0));

    #pragma omp parallel for schedule(static) num_threads(math::get_num_threads())
    for (int i = 0; i < m; ++i) {
        for (int p = 0; p < k; ++p) {
            T a_ip = A[i*k + p];
            const T* __restrict Bj = &B[p*n];
            T* __restrict Ci = &C[i*n];
            for (int j = 0; j < n; ++j) {
                Ci[j] += a_ip * Bj[j];
            }
        }
    }
    return C;
}

template <typename T>
inline std::vector<T> matmul(const std::vector<T>& A, const std::vector<T>& x, 
                             int m, int n){
    assert(A.size() == size_t(m)*size_t(n));
    assert(x.size() == size_t(n));
    return matmul(A, x, m, n, 1);
}

template<typename T>
inline std::vector<T> mul(const std::vector<T>& A, const std::vector<T>& B){
    assert(A.size() == B.size());
    std::vector<T> C(A.size());
    #pragma omp parallel for schedule(static) num_threads(math::get_num_threads())
    for(int i = 0; i < (int)A.size(); i++){
        C[i] = A[i] * B[i];
    }
    return C;
}

template<typename T>
inline std::vector<T> mul(const std::vector<T>& A, T scale){
    std::vector<T> B(A.size());
    #pragma omp parallel for schedule(static) num_threads(math::get_num_threads())
    for (int i = 0; i < (int)A.size(); ++i) B[i] = A[i] * scale;
    return B;
}

template<typename T>
inline std::vector<T> add(const std::vector<T>& A, const std::vector<T>& B){
    assert(A.size() == B.size());
    std::vector<T> C(A.size());
    #pragma omp parallel for schedule(static) num_threads(math::get_num_threads())
    for (int i = 0; i < (int)A.size(); ++i) C[i] = A[i] + B[i];
    return C;
}

template<typename T>
inline std::vector<T> sub(const std::vector<T>& A, const std::vector<T>& B){
    assert(A.size() == B.size());
    std::vector<T> C(A.size());
    #pragma omp parallel for schedule(static) num_threads(math::get_num_threads())
    for (int i = 0; i < (int)A.size(); ++i) C[i] = A[i] - B[i];
    return C;
}

template<typename T>
inline std::vector<T> transpose(const std::vector<T>& A, int m, int n){
    assert(A.size() == size_t(m)*size_t(n));
    std::vector<T> B(m*n);
    #pragma omp parallel for schedule(static) num_threads(math::get_num_threads())
    for(int i = 0; i < m; ++i){
        for(int j = 0; j < n; ++j){
            B[j*m + i] = A[i*n + j];
        }
    }
    return B;
}

// ---------- SCALAR OPERATIONS ----------
template <typename T>
inline T dot(const std::vector<T>& A, const std::vector<T>& B){
    assert(A.size() == B.size());
    T s = T(0);
    #pragma omp parallel for schedule(static) reduction(+:s) num_threads(math::get_num_threads())
    for (int i = 0; i < (int)A.size(); ++i) s += A[i] * B[i];
    return s;
}

template <typename T>
inline T dot(const std::vector<T>& A, const std::vector<T>& B, 
             const std::vector<T>& Mat){
    assert(A.size() == B.size());
    assert(Mat.size() == A.size()*A.size());
    std::vector<T> C = matmul(Mat, B, (int)B.size(), (int)B.size());
    return dot(A, C);
}

template <typename T>
inline T norm(const std::vector<T>& A, bool inf = false){
    if (inf) {
        T mx = T(0);
        #pragma omp parallel for schedule(static) reduction(max:mx) num_threads(math::get_num_threads())
        for (int i = 0; i < (int)A.size(); ++i) {
            T av = static_cast<T>(std::abs(A[i]));
            if (av > mx) mx = av;
        }
        return mx;
    } else {
        long double s = 0.0L;
        #pragma omp parallel for schedule(static) reduction(+:s) num_threads(math::get_num_threads())
        for (int i = 0; i < (int)A.size(); ++i) {
            long double v = static_cast<long double>(A[i]);
            s += v * v;
        }
        return static_cast<T>(std::sqrt(s));
    }
}

template <typename T>
inline bool symmetric(const std::vector<T>& A, int n){
    assert(A.size() == size_t(n)*size_t(n));
    bool ok = true;
    #pragma omp parallel for schedule(static) reduction(&&:ok) num_threads(math::get_num_threads())
    for(int i = 0; i < n; ++i){
        for(int j = i+1; j < n; ++j){
            T Aij = A[i*n + j];
            T Aji = A[j*n + i];
            ok = ok && (std::abs(Aij - Aji) <= static_cast<T>(1e-8));
        }
    }
    return ok;
}

} // namespace math
