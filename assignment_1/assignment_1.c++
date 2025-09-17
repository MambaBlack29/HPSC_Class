#include <bits/stdc++.h>
using namespace std;

struct Problem {
    int n = 0;
    vector<double> A; // column-major n x n
    vector<double> vec; // length n

    inline double Aat(int i, int j) const {
        // Column-major (MATLAB-like)
        return A[ size_t(i) + size_t(j) * size_t(n) ];
        // If your file is actually row-major, use:
        // return A[ size_t(j) + size_t(i) * size_t(n) ];
    }
};

static bool read_vector_file(const string& path, vector<double>& out) {
    ifstream in(path);
    if (!in) return false;
    out.clear();
    out.reserve(1<<20);
    double x;
    while (in >> x) out.push_back(x);
    return true;
}

static bool load_problem(const string& mat_path, const string& vec_path, Problem& P) {
    vector<double> flat, v;
    if (!read_vector_file(mat_path, flat)) return false;
    if (!read_vector_file(vec_path,  v))   return false;

    long long mlen = (long long)flat.size();
    int n = (int) llround(sqrt((long double)mlen));
    if (1LL * n * n != mlen || (int)v.size() != n) return false;

    P.n = n;
    P.A = std::move(flat);
    P.vec = std::move(v);
    return true;
}

// y = Gs * x   using t as scratch
static inline void apply_Gs(const Problem& P, double w,
                            const double* __restrict x,
                            double* __restrict y,
                            double* __restrict t) {
    const int n = P.n;
    // t = ((1-w)*D - w*U) * x
    for (int i = 0; i < n; ++i) {
        double di = P.Aat(i, i);
        double sum = (1.0 - w) * di * x[i];
        // subtract w * U(i,j) * x[j] for j>i
        for (int j = i+1; j < n; ++j) {
            sum -= w * P.Aat(i, j) * x[j];
        }
        t[i] = sum;
    }
    // solve (D + wL) y = t   (forward substitution)
    for (int i = 0; i < n; ++i) {
        double s = t[i];
        for (int j = 0; j < i; ++j) {
            s -= w * P.Aat(i, j) * y[j];
        }
        double di = P.Aat(i, i);
        y[i] = s / di;
    }
}

// c = w * (D + wL)^{-1} * vec
static inline void compute_c(const Problem& P, double w, vector<double>& c) {
    const int n = P.n;
    c.assign(n, 0.0);
    for (int i = 0; i < n; ++i) {
        double s = P.vec[i];
        for (int j = 0; j < i; ++j) {
            s -= w * P.Aat(i, j) * c[j];
        }
        double di = P.Aat(i, i);
        c[i] = s / di;
    }
    for (int i = 0; i < n; ++i) c[i] *= w;
}

// Power iteration (2-norm growth) to estimate spectral radius of Gs
static double spectral_radius_estimate(const Problem& P, double w,
                                       int maxit = 10000, double tol = 1e-10) {
    const int n = P.n;
    vector<double> x(n), y(n), t(n);

    std::mt19937_64 rng(12345);
    std::normal_distribution<double> nd(0.0, 1.0);
    for (int i = 0; i < n; ++i) x[i] = nd(rng);

    auto nrm2 = [&](const vector<double>& z)->double{
        long double s = 0.0L;
        for (double zz : z) s += (long double)zz * (long double)zz;
        return sqrt((double)s);
    };

    double lam_old = 0.0;
    for (int k = 0; k < maxit; ++k) {
        apply_Gs(P, w, x.data(), y.data(), t.data());
        double ny = nrm2(y);
        if (ny == 0.0) return 0.0;
        double nx = nrm2(x);
        double lam = ny / (nx > 0 ? nx : 1.0);

        // normalize y -> x
        for (int i = 0; i < n; ++i) x[i] = y[i] / ny;

        if (fabs(lam - lam_old) <= tol * max(1.0, fabs(lam))) {
            return fabs(lam);
        }
        lam_old = lam;
    }
    return fabs(lam_old);
}

// Fixed-point: x_{k+1} = Gs x_k + c  with ||x^{k+1}-x^k||_inf <= 1e-8*(1+||x^{k+1}||_inf)
static long run_fp(const Problem& P, double w, const vector<double>& c, vector<double>* x0 = nullptr) {
    const double tol = 1e-8;
    const long long maxit = (long long)1e8;
    const int n = P.n;

    vector<double> x(n), xnew(n), t(n);
    if (x0 && (int)x0->size() == n) {
        x = *x0;
    } else {
        std::random_device rd;
        std::mt19937_64 rng(rd());
        std::normal_distribution<double> nd(0.0, 1.0);
        for (int i = 0; i < n; ++i) x[i] = nd(rng);
    }

    auto norm_inf = [&](const vector<double>& z)->double{
        double m = 0.0;
        for (double v : z) m = max(m, fabs(v));
        return m;
    };

    long long k = 0;
    for (k = 1; k <= maxit; ++k) {
        apply_Gs(P, w, x.data(), xnew.data(), t.data());
        for (int i = 0; i < n; ++i) xnew[i] += c[i];

        double diff_inf = 0.0, xnew_inf = 0.0;
        for (int i = 0; i < n; ++i) {
            diff_inf = max(diff_inf, fabs(xnew[i] - x[i]));
            xnew_inf = max(xnew_inf, fabs(xnew[i]));
        }
        if (diff_inf <= tol * (1.0 + xnew_inf)) break;
        x.swap(xnew);
    }
    if (k > maxit) return (long)maxit;
    return (long)k;
}

int main(int argc, char** argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " Kmat.txt Fvec.txt\n";
        return 1;
    }
    Problem P;
    if (!load_problem(argv[1], argv[2], P)) {
        cerr << "Failed to load inputs (shapes/format mismatch?)\n";
        return 2;
    }

    const double w = 1.9;

    vector<double> c;
    compute_c(P, w, c);

    double specRad = spectral_radius_estimate(P, w, 10000, 1e-10);
    double convRate = (specRad > 0.0) ? -log(specRad) : numeric_limits<double>::infinity();

    const int num_iter = 3000;
    long double sm_acc = 0.0L;
    for (int i = 0; i < num_iter; ++i) {
        sm_acc += (long double) run_fp(P, w, c, nullptr);
    }
    double sm = (double)(sm_acc / (long double)num_iter);

    cout.setf(std::ios::fixed); cout << setprecision(15);
    cout << "specRad = "  << specRad  << "\n";
    cout << "convRate = " << convRate << "\n";
    cout << "sm = "       << sm       << "\n";
    return 0;
}
