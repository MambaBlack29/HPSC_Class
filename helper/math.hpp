#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <cassert>

namespace math{

// CONSTANT MATRICES
template<typename T>
inline std::vector<T> identity(int n){
    std::vector<T> A(n*n, T(0));
    for(int i = 0; i < n; ++i) A[i*n + i] = T(1);
    return A;
}

template<typename T>
inline std::vector<T> ones(int n, T val = T(1)){
    return std::vector<T>(n*n, val);
}

template <typename T>
inline std::vector<T> diag(const std::vector<T>& A, int n, bool vec = false){
    assert(A.size() == n*n);
    std::vector<T> B;
    if(vec) B.assign(n, T(0));
    else B.assign(n*n, T(0));
    for(int i = 0; i < n; i++){
        if(vec) B[i] = A[i*n + i];
        else B[i*n + i] = A[i*n + i];
    }
    return B;
}

template <typename T>
inline std::vector<T> upper(const std::vector<T>& A, int n, int offset = 1){
    assert(A.size() == n*n);
    std::vector<T> B(n*n, 0);
    for(int i = 0; i < n; ++i){
        for(int j = i+offset; j < n; ++j){
            B[i*n + j] = A[i*n + j];
        }
    }
    return B;
}

template <typename T>
inline std::vector<T> lower(const std::vector<T>& A, int n, int offset = 1){
    assert(A.size() == n*n);
    std::vector<T> B(n*n, 0);
    for(int i = 0; i < n; ++i){
        for(int j = 0; j <= i-offset; ++j){
            B[i*n + j] = A[i*n + j];
        }
    }
    return B;
}

// MATRIX OPERATIONS
template <typename T>
inline std::vector<T> matmul(const std::vector<T>& A, const std::vector<T>& B, 
                             int m, int k, int n){
    assert(A.size() == m*k);
    assert(B.size() == k*n);
    std::vector<T> C(m*n, T(0));
    for (int i = 0; i < m; ++i) {
        for (int p = 0; p < k; ++p) {
            T a_ip = A[i*k + p];
            for (int j = 0; j < n; ++j) {
                C[i*n + j] += a_ip * B[p*n + j];
            }
        }
    }
    return C;
}

template <typename T>
inline std::vector<T> matmul(const std::vector<T>& A, const std::vector<T>& x, 
                             int m, int n){
    assert(A.size() == m*n);
    assert(x.size() == n);
    return matmul(A, x, m, n, 1);
}

template<typename T>
inline std::vector<T> mul(const std::vector<T>& A, const std::vector<T>& B){
    assert(A.size() == B.size());
    std::vector<T> C(A.size());
    for(int i = 0; i < A.size(); i++){
        C[i] = A[i] * B[i];
    }
    return C;
}

template<typename T>
inline std::vector<T> mul(const std::vector<T>& A, T scale){
    std::vector<T> B = A;
    for (auto& v : B) v *= scale;
    return B;
}

template<typename T>
inline std::vector<T> add(const std::vector<T>& A, const std::vector<T>& B){
    assert(A.size() == B.size());
    std::vector<T> C(A.size());
    for (int i = 0; i < A.size(); ++i) C[i] = A[i] + B[i];
    return C;
}

template<typename T>
inline std::vector<T> sub(const std::vector<T>& A, const std::vector<T>& B){
    assert(A.size() == B.size());
    std::vector<T> C(A.size());
    for (int i = 0; i < A.size(); ++i) C[i] = A[i] - B[i];
    return C;
}

template<typename T>
inline std::vector<T> transpose(const std::vector<T>& A, int m, int n){
    assert(A.size() == m*n);
    std::vector<T> B(m*n);
    for(int i = 0; i < m; ++i){
        for(int j = 0; j < n; ++j){
            B[j*m + i] = A[i*n + j];
        }
    }
    return B;
}

// SCALAR OPERATIONS
template <typename T>
inline T dot(const std::vector<T>& A, const std::vector<T>& B){
    assert(A.size() == B.size());
    T s = T(0);
    for (int i = 0; i < A.size(); ++i) s += A[i] * B[i];
    return s;
}

template <typename T>
inline T dot(const std::vector<T>& A, const std::vector<T>& B, 
             const std::vector<T>& Mat){
    assert(A.size() == B.size());
    assert(Mat.size() == A.size()*A.size());
    std::vector<T> C = matmul(Mat, B, B.size(), B.size());
    return dot(A, C);
}

template <typename T>
inline T norm(const std::vector<T>& A, bool inf = false){
    if (inf) {
        T mx = T(0);
        for (const auto& v : A) mx = std::max(mx, static_cast<T>(std::abs(v)));
        return mx;
    } else {
        long double s = 0;
        for (const auto& v : A) s += static_cast<long double>(v) * v;
        return std::sqrt(s); 
    }
}

template <typename T>
inline bool symmetric(const std::vector<T>& A, int n){
    assert(A.size() == n*n);
    for(int i = 0; i < n; ++i){
        for(int j = i+1; j < n; ++j){
            T Aij = A[i*n + j];
            T Aji = A[j*n + i];
            if(std::abs(Aij - Aji) > 1e-8) return false;
        }
    }
    return true;
}

}