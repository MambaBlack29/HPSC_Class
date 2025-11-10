# Assignment 3 — BiCGSTAB Parallel Scaling
**Name:** Aryan Agarwal  
**Roll No.:** 22EC39006  

### Experimental Setup
- `.msh` files generated using `gmsh` software and converted to `fem.c` readable format using the given `C` code files
- `fem.c` updated for the given equation and boundary conditions and run for meshes with around $100$, $200$ and $400$ points
- `math_omp.hpp` built as a lightweight linear algebra library to support **parallelised** vector and matrix operations required for the solvers
- `iter.hpp` contains the BiCGSTAB implementation used in all runs.  
- `main.cpp` wires everything together and benchmarks BiCGSTAB over multiple grid sizes and thread counts.  
- Initial guess \(x_0=\text{rand\_vec}\) for all runs; stopping tolerance fixed at \(10^{-8}\) on the residual norm.  
- Timings report solver wall-time; iteration counts are the number of BiCGSTAB outer iterations to convergence; both taken as average over 1000 runs per experiment

---

### Biconjugate Gradient Stabilized (BiCGSTAB)
Given \(A\in\mathbb{R}^{n\times n}\), \(b\in\mathbb{R}^{n}\):
1. Choose initial guess $x^{(0)}$; compute initial residual $r^{(0)} = b - A x^{(0)}$  
2. Set $r^{*} = r^{(0)}$, $p^{(0)} = r^{(0)}$  
3. For $k = 0, 1, 2, \dots$ until convergence Do:  
    1. Compute $A p^{(k)}$ and step length  
    $$
    \alpha_k = \frac{r^{(k)T} r^{*}}{(A p^{(k)})^T r^{*}}
    $$  
    2. Compute intermediate residual  
    $$
    s^{(k)} = r^{(k)} - \alpha_k (A p^{(k)})
    $$  
    3. Compute $A s^{(k)}$ and $\omega_k$  
    $$
    \omega_k = \frac{(A s^{(k)})^T s^{(k)}}{(A s^{(k)})^T (A s^{(k)})}
    $$  
    4. Update solution  
    $$
    x^{(k+1)} = x^{(k)} + \alpha_k p^{(k)} + \omega_k s^{(k)}
    $$  
    5. Update residual  
    $$
    r^{(k+1)} = s^{(k)} - \omega_k (A s^{(k)})
    $$  
    6. Compute $\beta_k$  
    $$
    \beta_k = \frac{r^{(k+1)T} r^{*}}{r^{(k)T} r^{*}} \cdot \frac{\alpha_k}{\omega_k}
    $$  
    7. Update search direction  
    $$
    p^{(k+1)} = r^{(k+1)} + \beta_k \big(p^{(k)} - \omega_k (A p^{(k)})\big)
    $$  
    8. Check convergence: 
    $$
    \text{if }\, \|r^{(k+1)}\|_2 \leq \text{tol}\cdot\|b\|_2 \;\text{then Stop}  
    $$
4. EndDo  

### BiCGSTAB Results (Time, Iterations, Speedup & Efficiency)

> **Notes:**  
> Speedup \(S_p = T_{1}/T_{p}\). Efficiency \(E_p = S_p/p\).  
> Times are in **milliseconds** (ms); for the smallest case they are in **microsecond** (μs).

#### Grid: **n = 400 (418 points)**
| Threads | Time (ms) | Iters | Speedup | Efficiency |
|:------:|:---------:|:-----:|:-------:|:----------:|
| 1 (serial) | 12.61 | 12.06 | 1.00 | 1.00 |
| 2 | 6.86 | 12.91 | 1.84 | 0.92 |
| 4 | 3.51 | 12.23 | 3.59 | 0.90 |
| 8 | 2.15 | 12.38 | 5.87 | 0.73 |

#### Grid: **n = 200 (198 points)**
| Threads | Time (ms) | Iters | Speedup | Efficiency |
|:------:|:---------:|:-----:|:-------:|:----------:|
| 1 (serial) | 2.52 | 10.48 | 1.00 | 1.00 |
| 2 | 1.36 | 10.43 | 1.85 | 0.93 |
| 4 | 0.80 | 10.66 | 3.15 | 0.79 |
| 8 | 0.61 | 10.95 | 4.13 | 0.52 |

#### Grid: **n = 100 (95 points)**
| Threads | Time (μs) | Iters | Speedup | Efficiency |
|:------:|:---------:|:-----:|:-------:|:----------:|
| 1 (serial) | 600.29 | 10.67 | 1.00 | 1.00 |
| 2 | 435.16 | 11.04 | 1.38 | 0.69 |
| 4 | 302.34 | 10.81 | 1.99 | 0.50 |
| 8 | 316.16 | 10.75 | 1.90 | 0.24 |

---

### Observations
- **Iterations are stable** across thread counts (variation \(\lesssim\) 1 iteration), as expected—parallelization changes runtime, not convergence behavior.  
- **Near-linear scaling up to 4 threads** for larger problems (n = 400/418 and 200/198).  
- **Diminishing returns at 8 threads**, likely due to memory bandwidth limits and synchronization overheads typical of sparse mat-vec workloads.  
- **Small problem sizes saturate early**: for n = 100 (95 points), overhead dominates beyond 4 threads (8-thread run is slightly slower than 4-thread).  

---

### Takeaways
- For **production runs on larger grids**, 4 threads offer an excellent **speedup/efficiency** balance.  
- Moving from 4 → 8 threads helps on the largest grid, but with reduced efficiency—consider it only if cores are otherwise idle.  
- For **small grids**, prefer fewer threads; the parallel overhead can erase gains.
